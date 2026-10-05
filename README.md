# Banking Transactions: Risk, Compliance & Fraud Analysis 🏦

Analysis of 52,000+ banking transactions (2022–2025) covering account activity,
transaction behavior, AML/KYC compliance patterns, and an exploratory machine
learning model for fraud detection.

## 📌 Objective
- Clean and standardize a large financial transactions dataset with realistic data-quality issues
- Explore transaction patterns across account types, channels, currencies, and branches
- Investigate risk scores, fees, and AML/KYC compliance status
- Attempt to build a predictive fraud detection model and critically evaluate its performance

## 🛠️ Tools & Libraries
- **Python** — Pandas, NumPy, Matplotlib, Seaborn
- **Scikit-learn** — `StandardScaler`, `RandomForestClassifier`, `train_test_split`, classification metrics

## 🧹 Data Cleaning & Feature Engineering
- Standardized inconsistent text casing across `account_status`, `aml_check_status`, `kyc_status`, `payment_method`
- Converted negative transaction amounts to absolute values for consistent analysis
- Filled missing `merchant_name` with `"Unknown"` rather than dropping transactions
- Imputed missing `credit_score` using a layered approach: first forward/backward-filled **per customer**, then filled remaining gaps using the median **per customer segment + fraud flag group**
- Added a `Credit_score_missing` flag to preserve the information that a score was originally absent, in case missingness itself is meaningful
- Parsed `transaction_date` and `created_at` into proper datetime types
- Removed duplicate records

## 📊 Key Findings (EDA)

| Question | Finding |
|---|---|
| Account activity | Active accounts far outnumber Frozen, Closed, and Dormant combined |
| Transaction volume trend | Peaked in 2023, declined through 2024–2025 |
| Currency | Fairly even split across PLN, USD, EUR, CHF, GBP — EUR slightly leads by amount |
| International transactions | Nearly a 50/50 split between domestic and international |
| Dominant payment method | **SEPA**, both by transaction count and total amount |
| Fees | Fairly uniform across transaction type, channel, branch, payment method, and customer segment no single factor stands out |
| AML status | Majority of checks **Passed**; remainder split across Failed, Pending, Manual Review, and Skipped |
| KYC status | **Verified** is the most common status, followed by roughly equal shares of Pending, Expired, and Not Required |
| Correlation between numeric features | Amount, balance, fee, risk score, and credit score show **no meaningful linear correlation** with each other |

## 🤖 Fraud Detection Model (Exploratory)

Built a `RandomForestClassifier` to predict `fraud_flag` using all available transaction
and account features (one-hot encoded categoricals, scaled numericals, 80/20 train-test split).

**Results:**
| Metric | Value |
|---|---|
| Accuracy | 0.50 |
| ROC-AUC | 0.498 |
| Precision / Recall (fraud class) | 0.50 / 0.47 |

**Honest interpretation:** An ROC-AUC of ~0.50 means the model performs no better
than random guessing. This indicates that, within this dataset, `fraud_flag` does
not carry a learnable signal from the available features consistent with the
correlation heatmap, which showed no strong linear relationships between numeric
fields. Rather than overstate the model's performance, this result is reported as-is.

**What this suggests for next steps:**
- The fraud label may be randomly assigned in this dataset (common in synthetic data)
- Real fraud detection typically requires behavioral/sequential features (e.g. transaction velocity, deviation from a customer's historical pattern) rather than single-transaction snapshots
- Class imbalance handling (SMOTE, class weighting) and alternative models (XGBoost, anomaly detection) would be reasonable next experiments if the label does carry real signal

## 🚀 How to Run
```bash
git clone https://github.com/jobinjosej253/banking-fraud-risk-analysis.git
cd banking-fraud-risk-analysis
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebooks/banking_dataanalysis.ipynb
```

## 📂 Repo Structure
```
├── finance_banking_transactions_2022_2025.csv
├── banking_dataanalysis.ipynb
└── README.md
```

## 📄 License
MIT
