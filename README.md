# 💳 Anomaly Detection in Credit Card Transactions

Detecting fraudulent credit card transactions with **unsupervised learning only**. The fraud labels are hidden
from every model and used only at the end to evaluate the results.

📓 **Main notebook:** [`anomaly_detection.ipynb`](anomaly_detection.ipynb)

## Dataset
[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 transactions over two days, of which only **492 (0.17 %) are frauds**. The features `V1`–`V28` are anonymised
PCA components, plus `Time` and `Amount`.
The CSV (~150 MB) is not stored in this repo. The notebook downloads it automatically on the first run.

## Pipeline
1. **Data exploration**: `Amount` / `Time` distributions, class imbalance, day/night patterns, heavy-tailed features
2. **Preprocessing**: labels hidden, cyclic hour-of-day features, log-amount, `StandardScaler` (and why not `MinMaxScaler`), PCA & t-SNE visualisation
3. **Models**: distance baseline, **Isolation Forest**, **One-Class SVM**, **Autoencoder (PyTorch)**, rank-average ensemble
4. **Evaluation**: precision, recall, F1, PR-AUC, ROC-AUC at several alert budgets
5. **Insights**: what gets flagged, false alarms vs missed frauds, model agreement

## Key results (test set: 85,443 transactions, 136 frauds)

| Model | PR-AUC | ROC-AUC | Precision @ 0.2 % | Recall @ 0.2 % | F1 @ 0.2 % |
|---|---|---|---|---|---|
| **Isolation Forest** | **0.253** | **0.961** | **0.280** | **0.360** | **0.315** |
| Distance baseline | 0.200 | 0.961 | 0.269 | 0.309 | 0.288 |
| One-Class SVM | 0.125 | 0.956 | 0.136 | 0.184 | 0.156 |
| Ensemble | 0.076 | 0.943 | 0.089 | 0.110 | 0.099 |
| Autoencoder | 0.035 | 0.899 | 0.077 | 0.088 | 0.082 |
| *Random guess* | *0.002* | *0.500* | | | |

*"@ 0.2 %" = flag the 0.2 % most anomalous transactions for review.*

- Isolation Forest is **~160× better than random** at ranking frauds. Flagging 1 % of transactions catches ~60 % of frauds, and 5 % catches ~87 %.
- **Caught frauds**: small amounts (often < €2, which suggests card testing), often at night, with extreme values in `V17`, `V14`, `V12`, `V4`, `V11`.
- **False alarms**: mostly large legitimate purchases. The models detect *unusual*, not necessarily *fraudulent*.
- **Missed frauds** (~60 %) look like ordinary purchases. This is the natural limit of a purely unsupervised approach.

## How to run
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter notebook anomaly_detection.ipynb
```
A full run takes about 6-8 minutes on a laptop CPU (t-SNE and the autoencoder take the longest).

## Project structure
```
anomaly-detection/
├── anomaly_detection.ipynb   # the full analysis (with outputs)
├── requirements.txt          # pinned dependencies
├── README.md
├── .gitignore                # excludes .venv/ and data/
└── data/                     # created automatically; holds creditcard.csv (not committed)
```
