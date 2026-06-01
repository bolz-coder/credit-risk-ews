# ============================================================
# CREDIT RISK EARLY WARNING SYSTEM
# Nigerian Bank Loan Portfolio — ML Classification
# ============================================================
# Dataset: Bolanle.xlsx (company loan portfolio, Ogun State)
# Target: 3-class risk model (Low / Medium / High Risk)
# Model: XGBoost (primary) vs Logistic Regression, Random Forest
# ============================================================

# ── 0. INSTALL & IMPORTS ─────────────────────────────────
# Uncomment the line below if running in Google Colab
# !pip install xgboost lightgbm shap imbalanced-learn openpyxl -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import pickle
import json

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

COLORS = {
    'low':    '#1D9E75',
    'medium': '#EF9F27',
    'high':   '#D85A30',
    'blue':   '#185FA5',
    'gray':   '#888780'
}

# ── 1. LOAD DATA ─────────────────────────────────────────
# Change path if running in Colab after uploading the file
df = pd.read_excel('Bolanle.xlsx')
print(f"Dataset shape: {df.shape}")
print(f"\nClassification distribution:\n{df['Classification'].value_counts()}")

# ── 2. TARGET VARIABLE ───────────────────────────────────
# 3-class ordinal risk model:
# 0 = Low Risk    → Performing (2,414 loans)
# 1 = Medium Risk → Watchlist 1, 2, 3 (875 loans)
# 2 = High Risk   → Substandard, Loss, Doubtful (220 loans)

def assign_risk(classification):
    if classification == 'Performing':
        return 0
    elif classification in ['Watchlist 1', 'Watchlist 2', 'Watchlist 3']:
        return 1
    else:  # Substandard, Loss, Doubtful
        return 2

df['RiskLevel'] = df['Classification'].apply(assign_risk)

print(f"\nRisk Level distribution:")
print(df['RiskLevel'].value_counts().rename({0:'Low Risk', 1:'Medium Risk', 2:'High Risk'}))
print(f"\nClass balance: {df['RiskLevel'].value_counts().to_dict()}")


# ── 3. FEATURE ENGINEERING ───────────────────────────────
# CRITICAL: Exclude leakage columns (derived AFTER classification):
# PDOs, VAR, Mths in Delinquent, Amt in Delinquent, Diff
# Also exclude: ContractId, CustomerId, ClientEmployeeNumber (identifiers)
# Also exclude: State (all Ogun — no variance), Categorisation (derived target)
# Also exclude: Classification (the target itself)
# Exclude LastRepaymentPostedDate / LastRepaymentPosted (1128 missing, leakage risk)

today = pd.Timestamp.today()

def feature_engineer(df):
    df = df.copy()

    # ── Age from DateofBirth ──
    df['DateofBirth'] = pd.to_datetime(df['DateofBirth'])
    df['Age'] = ((today - df['DateofBirth']).dt.days / 365.25).astype(int)

    # ── Loan tenure features ──
    df['CreationDate']      = pd.to_datetime(df['CreationDate'])
    df['MaturityDate']      = pd.to_datetime(df['MaturityDate'])
    df['FirstRepaymentDate']= pd.to_datetime(df['FirstRepaymentDate'])

    df['LoanAgeMonths']     = df['MonthsTillDate']                         # months since creation
    df['RemainingMonths']   = df['MonthsTillMaturity']                     # months till maturity
    df['ProgressRatio']     = df['LoanAgeMonths'] / df['Tenor'].replace(0, np.nan)  # how far through loan

    # ── Financial ratios ──
    df['TotalInterest']         = df['InterestTillDate']
    df['InterestToLoan']        = df['InterestTillDate'] / df['LoanAmount'].replace(0, np.nan)
    df['LoanValueToAmount']     = df['LoanValue'] / df['LoanAmount'].replace(0, np.nan)
    df['TotalCollectableRatio'] = df['Total Collectable'] / df['LoanAmount'].replace(0, np.nan)
    df['MonthlyBurden']         = df['LoanMonthlyInstallment'] / df['LoanAmount'].replace(0, np.nan)
    df['MaintenanceFeeRatio']   = df['Cummulative Maintenance Fee'] / df['LoanAmount'].replace(0, np.nan)
    df['DeductionsRatio']       = df['DeductionsTillDate'] / df['Total Collectable'].replace(0, np.nan)
    df['ActualVsExpected']      = df['ActualDTD'] / df['DeductionsTillDate'].replace(0, np.nan)

    # ── Log transforms for skewed amounts ──
    df['LoanAmount_log']        = np.log1p(df['LoanAmount'])
    df['LoanValue_log']         = np.log1p(df['LoanValue'])
    df['Installment_log']       = np.log1p(df['LoanMonthlyInstallment'])

    # ── Categorical encodings ──
    df['IsMale']       = (df['Gender'] == 'Male').astype(int)
    df['Gender_filled']= df['Gender'].fillna('Unknown')
    df['IsTopup']      = (df['LoanCategory'] == 'TOPUP').astype(int)

    return df

df_feat = feature_engineer(df)

# ── Define feature set (no leakage) ──
FEATURE_COLS = [
    # Borrower
    'Age', 'IsMale',
    # Loan terms
    'LoanAmount', 'LoanAmount_log', 'Tenor', 'LoanMonthlyInstallment',
    'Installment_log', 'IsTopup',
    # Rates
    'ContractualRate', 'Rate', 'ManagementFee', 'Monthly Maintenance Fee',
    # Portfolio position
    'LoanAgeMonths', 'RemainingMonths', 'ProgressRatio',
    # Financial ratios
    'InterestToLoan', 'LoanValueToAmount', 'TotalCollectableRatio',
    'MonthlyBurden', 'MaintenanceFeeRatio', 'DeductionsRatio',
    'LoanValue', 'LoanValue_log', 'Total Collectable',
    'InterestTillDate', 'TotalInterest',
    'DeductionsTillDate', 'ActualDTD', 'ActualVsExpected',
    'Cummulative Maintenance Fee',
]

X = df_feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_feat['RiskLevel']

print(f"\n✓ Feature matrix: {X.shape}")
print(f"Features used: {FEATURE_COLS}")


# ── 4. EXPLORATORY DATA ANALYSIS ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Credit Risk EDA — Nigerian Bank Loan Portfolio", fontsize=14, fontweight='bold')

risk_labels = {0: 'Low Risk\n(Performing)', 1: 'Medium Risk\n(Watchlist)', 2: 'High Risk\n(Substandard+)'}
risk_colors = [COLORS['low'], COLORS['medium'], COLORS['high']]

# 4a. Class distribution
ax = axes[0, 0]
counts = df['RiskLevel'].value_counts().sort_index()
bars = ax.bar([risk_labels[i] for i in counts.index], counts.values,
               color=risk_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}\n({val/len(df):.1%})', ha='center', fontsize=10)
ax.set_title("Risk Level Distribution")
ax.set_ylabel("Count")

# 4b. Loan amount by risk
ax = axes[0, 1]
for i, (risk, color) in enumerate(zip([0,1,2], risk_colors)):
    vals = df_feat[df_feat['RiskLevel'] == risk]['LoanAmount'] / 1000
    ax.hist(vals.clip(upper=vals.quantile(0.99)), bins=30, alpha=0.6,
            label=list(risk_labels.values())[i], color=color, density=True)
ax.set_title("Loan Amount Distribution by Risk")
ax.set_xlabel("Loan Amount (₦ '000)")
ax.legend(fontsize=8)

# 4c. Age distribution by risk
ax = axes[0, 2]
for i, (risk, color) in enumerate(zip([0,1,2], risk_colors)):
    vals = df_feat[df_feat['RiskLevel'] == risk]['Age']
    ax.hist(vals, bins=25, alpha=0.6,
            label=list(risk_labels.values())[i], color=color, density=True)
ax.set_title("Age Distribution by Risk Level")
ax.set_xlabel("Age (years)")
ax.legend(fontsize=8)

# 4d. Gender vs risk
ax = axes[1, 0]
gender_risk = df_feat.groupby('Gender_filled')['RiskLevel'].value_counts(normalize=True).unstack().fillna(0)
gender_risk[[0,1,2]].plot(kind='bar', ax=ax, color=risk_colors, edgecolor='white', width=0.6)
ax.set_title("Risk Distribution by Gender")
ax.set_xlabel("")
ax.set_ylabel("Proportion")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Low', 'Medium', 'High'], fontsize=9)

# 4e. Loan category vs risk
ax = axes[1, 1]
cat_risk = df_feat.groupby('LoanCategory')['RiskLevel'].value_counts(normalize=True).unstack().fillna(0)
cat_risk[[0,1,2]].plot(kind='bar', ax=ax, color=risk_colors, edgecolor='white', width=0.5)
ax.set_title("Risk Distribution by Loan Category")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Low', 'Medium', 'High'], fontsize=9)

# 4f. Tenor vs risk
ax = axes[1, 2]
tenor_risk = df_feat.groupby('Tenor')['RiskLevel'].mean()
ax.bar(tenor_risk.index, tenor_risk.values, color=COLORS['blue'], edgecolor='white', width=0.7)
ax.set_title("Avg Risk Score by Loan Tenor (months)")
ax.set_xlabel("Tenor (months)")
ax.set_ylabel("Mean Risk Level")

plt.tight_layout()
plt.savefig('01_eda.png', bbox_inches='tight')
plt.close()
print("✓ EDA saved → 01_eda.png")


# ── 5. TRAIN / TEST SPLIT ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train class counts: {dict(y_train.value_counts().sort_index())}")
print(f"Test class counts:  {dict(y_test.value_counts().sort_index())}")


# ── 6. HANDLE CLASS IMBALANCE WITH SMOTE ─────────────────
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\n✓ After SMOTE — Train: {X_train_sm.shape}")
print(f"Resampled class counts: {dict(pd.Series(y_train_sm).value_counts().sort_index())}")


# ── 7. MODEL TRAINING ────────────────────────────────────
MODELS = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000,
        random_state=42, C=0.1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        max_depth=8, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        num_class=3, objective='multi:softprob',
        random_state=42, eval_metric='mlogloss', verbosity=0
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Cross-Validation (5-fold, macro F1) ───────────")
for name, model in MODELS.items():
    scores = cross_val_score(model, X_train_sm, y_train_sm,
                             cv=cv, scoring='f1_macro', n_jobs=-1)
    results[name] = {'cv_f1_mean': scores.mean(), 'cv_f1_std': scores.std()}
    print(f"  {name:<22} F1 macro: {scores.mean():.4f} ± {scores.std():.4f}")


# ── 8. TEST SET EVALUATION ───────────────────────────────
print("\n── Test Set Evaluation ───────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Comparison — Credit Risk Classification", fontsize=14, fontweight='bold')

palette = ['#185FA5', '#1D9E75', '#D85A30']
risk_names = ['Low Risk', 'Medium Risk', 'High Risk']

for i, (name, model) in enumerate(MODELS.items()):
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    f1_mac = f1_score(y_test, y_pred, average='macro')
    f1_wt  = f1_score(y_test, y_pred, average='weighted')
    auc    = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    results[name]['test_f1_macro']    = f1_mac
    results[name]['test_f1_weighted'] = f1_wt
    results[name]['test_auc_ovr']     = auc
    results[name]['y_pred']           = y_pred
    results[name]['y_prob']           = y_prob

    print(f"\n  {name}")
    print(f"    F1 Macro: {f1_mac:.4f}  |  F1 Weighted: {f1_wt:.4f}  |  AUC (OvR): {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=risk_names))

# Confusion matrices
for i, (name, info) in enumerate(results.items()):
    ax = axes[i]
    cm = confusion_matrix(y_test, info['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low', 'Medium', 'High'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nF1={info['test_f1_macro']:.3f} | AUC={info['test_auc_ovr']:.3f}")

plt.tight_layout()
plt.savefig('02_model_comparison.png', bbox_inches='tight')
plt.close()
print("\n✓ Model comparison saved → 02_model_comparison.png")

best_name = max(results, key=lambda n: results[n]['test_f1_macro'])
print(f"\n✓ Best model: {best_name} (F1 macro: {results[best_name]['test_f1_macro']:.4f})")


# ── 9. SHAP FEATURE IMPORTANCE ───────────────────────────
print("\n── SHAP Analysis (XGBoost) ───────────────────────")
xgb_model = MODELS['XGBoost']
xgb_model.fit(X_train_sm, y_train_sm)

sample_idx = X_test.sample(min(300, len(X_test)), random_state=42).index
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test.loc[sample_idx])

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("SHAP Feature Importance by Risk Class", fontsize=14, fontweight='bold')

for cls_idx, (cls_name, color) in enumerate(zip(risk_names, risk_colors)):
    ax = axes[cls_idx]
    mean_shap = np.abs(shap_values[:, :, cls_idx]).mean(axis=0) if shap_values.ndim == 3 else np.abs(shap_values[cls_idx]).mean(axis=0)
    feat_imp = pd.Series(mean_shap, index=FEATURE_COLS).sort_values(ascending=True).tail(12)
    ax.barh(feat_imp.index, feat_imp.values, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f"{cls_name}")
    ax.set_xlabel("Mean |SHAP|")

plt.tight_layout()
plt.savefig('03_shap_importance.png', bbox_inches='tight')
plt.close()
print("✓ SHAP saved → 03_shap_importance.png")


# ── 10. SAVE MODEL & ARTIFACTS ───────────────────────────
# Save the best-performing model (XGBoost)
best_model = MODELS['XGBoost']

with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save feature list and metadata
meta = {
    'feature_cols': FEATURE_COLS,
    'risk_labels': {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'},
    'model_name': 'XGBoost',
    'test_f1_macro': round(results['XGBoost']['test_f1_macro'], 4),
    'test_auc_ovr':  round(results['XGBoost']['test_auc_ovr'], 4),
    'today_ref': str(today.date()),
}
with open('credit_risk_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n✓ Model saved → credit_risk_model.pkl")
print(f"✓ Metadata saved → credit_risk_meta.json")


# ── 11. RESULTS SUMMARY ──────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
summary = pd.DataFrame({
    'Model': list(results.keys()),
    'CV F1 (mean)': [f"{results[n]['cv_f1_mean']:.4f}" for n in results],
    'CV F1 (std)':  [f"±{results[n]['cv_f1_std']:.4f}"  for n in results],
    'Test F1 Macro':    [f"{results[n]['test_f1_macro']:.4f}"    for n in results],
    'Test F1 Weighted': [f"{results[n]['test_f1_weighted']:.4f}" for n in results],
    'AUC (OvR)':        [f"{results[n]['test_auc_ovr']:.4f}"     for n in results],
}).set_index('Model')
print(summary.to_string())
print(f"\n✓ Best model: {best_name}")
print("\nFiles generated:")
print("  01_eda.png, 02_model_comparison.png, 03_shap_importance.png")
print("  credit_risk_model.pkl, credit_risk_meta.json")
