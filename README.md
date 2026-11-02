# Predicting Review Helpfulness via Linguistic Specificity, Structural Complexity, and Epistemic Hedging Across Domains

**Anonymous ACL Submission**

## Project Overview
This repository contains the codebase for evaluating review helpfulness prediction using the Yelp Open Dataset and generalizing models zero-shot to Amazon product reviews. The project frames helpfulness prediction as a regression task predicting $\log(1 + \text{useful})$ to account for heavily skewed vote distributions.

We evaluate and compare three distinct model architectures:
1. **Metadata Baseline:** A Gradient Boosted Decision Tree trained on word count, star rating, review year, and business popularity controls.
2. **Embeddings Baseline:** A fine-tuned pretrained transformer (`roberta-base`) combined with review-level controls.
3. **Combined Model:** Integrates transformer embeddings with custom linguistic feature families (Specificity, Structural Composition, and Epistemic Hedging).

---

## Repository Structure
* `helper.py`: Shared utilities containing random seed initialization and evaluation metrics (`RMSE`, `MAE`, Spearman’s $\rho$).
* `baseline.py`: Implements Model 1 (Metadata GBDT baseline with business-grouped splitting and log target transformation).
* `timing_test.py`: Benchmarks Model 2 (`roberta-base` with controls comparing frozen vs. fine-tuned execution times and performance).
* `requirements.txt`: Python package dependencies.
* `setup_and_run.sh`: Automated setup script for creating a virtual environment, installing dependencies, and running models.

---

## Getting Started & Execution

### 1. Prerequisites
Ensure you have Python 3.11 installed along with an active virtual environment.

### 2. Automatic Setup & Execution
You can set up the environment, install all dependencies, and run the baseline and timing tests with a single command:
```bash
./setup_and_run.sh