# Credit default scoring

CodeAlpha ML task 1. Prepared for Mariam Ibrahim's review.

**Status:** Working review package, not yet submitted. Prepared with AI assistance. Mariam should run the project, review the choices and explain the results before it is represented as her completed internship work. No claim of independent authorship, internship completion or certification is made.

## What the project does

Predict next-month default (1), not a general credit score. The source contains Taiwanese clients from 2005. Financial history features and three row-wise engineered features are used. ID and direct demographics (sex, education, marital status, age) are excluded. This does not remove proxy bias.

The initial 30,000 rows lose 732 identical retained-feature/label rows; another 170 rows with contradictory labels for an identical retained financial profile are excluded. The final 29,098 rows have unique retained feature profiles. This cleaning choice changes the population and may remove legitimate clients with similar histories; it is an educational leakage-control choice, not a production data-cleaning recommendation. No missing cells are expected. The preprocessing pipeline still includes a training-fitted median imputer.

Five-fold stratified cross-validation on the 80% training partition selects between logistic regression and a random forest by ROC-AUC. Only the selected model and a majority baseline are evaluated on the 20% test partition. The 0.5 classification threshold is not tuned on the test set. Accuracy alone is misleading for the imbalanced outcome. The model misses many defaults, so ROC-AUC and recall must be read together. This is not suitable for lending decisions, and no fairness, calibration, legal or out-of-time validation is claimed.

## Run it

Python 3.10.12 was used for the recorded run. Install Python first, then run from this folder:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell, instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
python -m unittest test_project.py
```

The first run of credit/sales needs internet to download the source data. Subsequent runs use the local cache. Iris and breast-cancer data are bundled with scikit-learn. Results are written under `results/`. No credentials are required. Nothing is sent or published by the scripts. The notebook needs Jupyter (`pip install notebook`) only if you want to open it interactively.

## Recorded held-out results

Seed: 42. Training rows: 23,278. Test rows: 5,820.

| Model | accuracy | precision | recall | f1 | roc_auc | average_precision |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7842 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2158 |
| forest | 0.8117 | 0.5786 | 0.4689 | 0.5180 | 0.7650 | 0.5421 |


Binary classification precision, recall and F1 use class 1 (default or malignant). Iris metrics use macro averaging. Classification row indexes in `test_predictions.csv` refer to the cleaned analysis table, not the original file. Sales row indexes are original CSV row indexes. Exact train-only cross-validation means and standard deviations are in `results/metrics.json`. Data-quality counts are in `results/data_quality.json` for classification.

## Files

- `main.py`: complete source, including data loading and model evaluation.
- `walkthrough.ipynb`: executed notebook with explanation, code, recorded output and plots.
- `results/`: actual run metrics, test predictions and charts, not invented examples.
- `test_project.py`: automated output checks and a full rerun.
- `requirements.txt`: versions used in the verified run.

## Data and attribution

Source: https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients

Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. UCI provides the original dataset; no financial records from Mariam or anyone in her contacts are used. Download checksum is saved in `results/data_provenance.json`.

## Review before publishing

Explain the target, the split, why the baseline matters, what leakage means, and one limitation. Re-run the code and check the outputs. A LinkedIn video/post and the official form are still required by CodeAlpha; neither is included as a completed step here.
