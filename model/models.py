# models.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import tensorflow as tf
import xgboost as xgb


@dataclass
class ModelSpec:
    model_key: str
    params: Dict[str, Any]

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_tr: pd.DataFrame, y_tr: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X_te: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def evaluate(self, X_te: pd.DataFrame, y_te: pd.Series) -> Dict[str, Any]:
        pass

class LSTMModel(BaseModel):
    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec
        self.num_layers = model_spec.params.get("num_layers", 2)
        self.hidden_units = model_spec.params.get("hidden_units", 64)
        self.learning_rate = model_spec.params.get("learning_rate", 0.001)
        self.epochs = model_spec.params.get("epochs", 100)
        self.batch_size = model_spec.params.get("batch_size", 32)
        self.dropout = model_spec.params.get("dropout", 0.0)
        self.early_stopping_patience = model_spec.params.get("early_stopping_patience", None)
        self.reduce_lr_patience = model_spec.params.get("reduce_lr_patience", None)
        self.reduce_lr_factor = model_spec.params.get("reduce_lr_factor", 0.5)
        self.loss = model_spec.params.get("loss", "mae")
        self.pad_value = model_spec.params.get("pad_value", 0.0)
        self.verbose = model_spec.params.get("verbose", 0)

        # Placeholder for the actual LSTM model
        self.model = None


    def _build_model(self, X_tr: pd.DataFrame) -> None:
        
        # --- Add layers to the model ---
        #inputs = tf.keras.Input(shape=(X_tr.shape[1], X_tr.shape[2]))  # (T, F)
        inputs = tf.keras.Input(shape=(None, X_tr.shape[2]))  # (T, F), T is variable
        x = tf.keras.layers.Masking(mask_value=self.pad_value)(inputs)
        for _ in range(max(0, self.num_layers - 1)):
            x = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout)(x)
        x = tf.keras.layers.LSTM(self.hidden_units, return_sequences=False, dropout=self.dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation=None)(x)

        # --- Compile the model ---
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss=self.loss, metrics=[self.loss])
        
        self.model = model        


    def fit(self, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> None:
        if X_val is not None and X_val.shape[0] == 0:
            X_val = None
        if y_val is not None and y_val.shape[0] == 0:
            y_val = None
        
        # --- Get LSTM model architecture ---
        if self.model is None:
            self._build_model(X_tr)

        # --- Define training callbacks - --
        callbacks = []
        if self.early_stopping_patience is not None and X_val is not None and y_val is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.early_stopping_patience, restore_best_weights=True
            ))
        if self.reduce_lr_patience is not None and X_val is not None and y_val is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=self.reduce_lr_patience, factor=self.reduce_lr_factor, min_lr=1e-6
            ))
        
        # --- Train the model ---
        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks
        )
    
    def predict(self, X_te: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        predictions = self.model.predict(X_te, verbose=self.verbose)
        return pd.Series(predictions.flatten())
    

    def evaluate(self, X_te: pd.DataFrame, y_te: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        results = self.model.evaluate(X_te, y_te, verbose=self.verbose, return_dict=True)
        return results


class XGBoostModel(BaseModel):
    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec

        # Core XGBoost parameters
        self.n_estimators = model_spec.params.get("n_estimators", 300)
        self.learning_rate = model_spec.params.get("learning_rate", 0.05)
        self.max_depth = model_spec.params.get("max_depth", 6)
        self.subsample = model_spec.params.get("subsample", 0.8)
        self.colsample_bytree = model_spec.params.get("colsample_bytree", 0.8)
        self.min_child_weight = model_spec.params.get("min_child_weight", 1.0)
        self.reg_alpha = model_spec.params.get("reg_alpha", 0.0)
        self.reg_lambda = model_spec.params.get("reg_lambda", 1.0)

        # Objective / evaluation
        self.objective = model_spec.params.get("objective", "reg:squarederror")
        self.eval_metric = model_spec.params.get("eval_metric", "rmse")

        # Training control
        self.early_stopping_rounds = model_spec.params.get("early_stopping_rounds", None)
        self.verbose = model_spec.params.get("verbose", 0)

        self.model: Optional[xgb.XGBRegressor] = None

    def _build_model(self) -> None:
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            eval_metric=self.eval_metric,
            verbosity=self.verbose,
            n_jobs=-1
        )

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        if self.model is None:
            self._build_model()

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            #early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose
        )

    def predict(self, X_te: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        preds = self.model.predict(X_te)
        return pd.Series(preds, index=X_te.index)

    def evaluate(self, X_te: pd.DataFrame, y_te: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        preds = self.predict(X_te)

        #rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
        mae = float(np.mean(np.abs(preds - y_te)))

        return {
            #"rmse": rmse,
            "mae": mae
        }
