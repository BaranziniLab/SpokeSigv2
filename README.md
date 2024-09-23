# SpokeSigv2

# Knowledge graph-based integration of individual electronic health records and common genetic variants to predict disease risk

## Introduction

This project explores the use of Propagated Spoke Entry Vectors (PSEVs) and combination genetic with clinical data to enhance disease prediction models. By leveraging both genetic and clinical information, we aim to improve the accuracy and robustness of predictive models in healthcare.

## Propagated Spoke Entry Vectors

Propagated Spoke Entry Vectors (PSEVs) are a novel approach to representing genetic and clinical information in a format suitable for machine learning models. PSEVs are derived from biological knowledge graphs, specifically from SPOKE.

Key features of PSEVs:
- Capture complex relationships between genes and biological entities
- Provide a dense, fixed-length vector representation of genetic information
- Enable integration of diverse biological knowledge into predictive models

## Combining Genetic and Clinical Data

Our approach combines PSEVs (representing genetic data) with traditional clinical data to create more comprehensive and powerful predictive models. The combination process involves:

1. Normalizing both genetic PSEV and clinical PSEV
2. Applying different weights to genetic and clinical components
3. Exploring the optimal balance between genetic and clinical information

Benefits of this combined approach:
- Leverages complementary information from both data types
- Potentially improves prediction accuracy for complex diseases
- Allows for personalized risk assessment based on both genetic predisposition and clinical factors





## Results and Interpretation

After running the scripts, you'll find the following results:

- ROC curves for different genetic-clinical weight combinations
- Performance metrics (AUC, F1 score, balanced accuracy) for each weight
- Predictions and probabilities for the validation set

Interpret these results to:
- Determine the optimal balance between genetic and clinical data
- Assess the model's performance across different disease outcomes
- Identify potential biomarkers or risk factors


---

For more information about SPOKE and PSEVs, please visit [SPOKE's official website](https://spoke.ucsf.edu/).
