# Privacy Requirements and Design Patterns Data Repository

This repository contains a structured dataset featuring pairs of privacy requirements and privacy design patterns, alongside their relevance scores. The dataset is organized into five distinct folds, reflecting a stratified k-fold partitioning approach as detailed in our accompanying paper.

## Dataset Overview

Each fold in the dataset represents a combination of privacy requirements and design patterns that have been evaluated for relevance. The organization of this data facilitates the use of machine learning algorithms for predictive modeling and analysis.

## Preprocessing Script

Included in this repository is a Python script, `feature_creation.py`, which is used to transform the raw data into a feature-rich format suitable for machine learning. This script is an essential part of preparing the data for the subsequent training and testing phases.

## Learning-to-Rank Data

The output of the feature creation process is a set of files in the LightSVM format, which is designed for use with Learning-to-Rank (LTR) algorithms. Due to the extensive size of this data, we have made it available online for easy access.

## Accessing the Data

The LTR training and testing data can be accessed via the following link: [https://bit.ly/letor_priv](https://bit.ly/letor_priv). This link will direct users to an online storage location where the data can be downloaded.

## Citation

If you utilize this dataset for your research, please cite our paper as follows:

"soon"