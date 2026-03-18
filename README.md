# WARP: Worktime-Aware Representation of Time for Predictive Process Monitoring
Authors anonymized

This repository contains the implementation and experimental framework for the research paper "WARP: Worktime-Aware Representation of Time for Predictive Process Monitoring".

## Repository structure
├── data/               # Source for synthetic event logs
├── load/               # Utilities for initial event log ingestion
├── split/              # Utilities for partitioning event logs into training and testing subsets
├── prefix/             # Utilities for generating sequence prefixes
├── worktime_mapping/   # Implementation of identification of working intervals, and clocktime-to-worktime and worktime-to-clocktime mapping
├── feature/            # Implementation for engineering predictive features
├── transform/          # Implementation for generating feature vectors
├── model/              # Implementation of for predictive model architectures
├── experiment/         # Scripts for coordinating predictive workflow for evaluation experiments
├── artifacts/          # Storage for experiment results
├── runner.ipynb        # Central execution notebook for the pipeline
├── requirements.txt    # Python requirements for reproduction
└── README.md           # This file