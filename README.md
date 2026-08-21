# ACDI + Drift-Triage

Person-B research workspace for the ACDI + Drift-Triage conference-paper project.

## Ownership

This repository is the independent Person-B development workspace.

Person A works independently on the data/model side. The two workstreams
will be integrated later through explicitly defined software contracts.

## Person-B Scope

Person B is responsible for:

- Drift monitoring
- Drift/attack injection framework
- Triage decision logic
- Defense actions
- Evaluation of monitoring, triage, and recovery behavior

## Repository Structure

src/
- contracts/
- monitors/
- injection/
- triage/
- defense/
- evaluation/

tests/
- contracts/
- monitors/
- injection/
- triage/
- defense/
- evaluation/

configs/
scripts/
notebooks/
docs/

## Data and Artifact Policy

Large datasets, processed datasets, checkpoints, predictions, experiment
logs, drift logs, figures, and other generated research artifacts are stored
in the Person-B Google Drive workspace.

They are not stored directly in this Git repository.

## Reproducibility

Experiments should record:

- experiment configuration
- random seeds
- dataset/version information
- model configuration
- code revision
- artifact locations
- evaluation results

## Person-A Integration Boundary

The downstream Person-B pipeline will consume a stable model-output contract
from Person A.

The intended conceptual flow is:

Window -> PredictionOutput -> Drift Monitoring -> SignalReport -> Triage -> Defense

The exact schemas will be defined and frozen before downstream implementation.

## Project Status

Foundation and Colab recovery infrastructure verified.

Current phase: FOUNDATION

Next phase: SOURCE CONTROL / CONTRACT DESIGN
