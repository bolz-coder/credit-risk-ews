import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Credit Risk Early Warning System",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0d2137 0%, #1a4a7a 100%);
    padding: 2.2rem 2rem; border-radius: 16px;
    margin-bottom: 1.8rem; text-align: center; color: white;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0; color: white; }
.hero p  { font-size: 0.9rem; opacity: 0.8; margin-top: 0.4rem; color: white; }

.result-low {
    background: linear-gradient(135deg, #e6fffa, #c6f6d5);
    border: 2px solid #38a169; border-radius: 16px;
    padding: 1.8rem; text-align: center; margin-top: 1.5rem;
}
.result-medium {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #d97706; border-radius: 16px;
    padding: 1.8rem; text-align: center; margin-top: 1.5rem;
}
.result-high {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border: 2px solid #e53e3e; border-radius: 16px;
    padding: 1.8rem; text-align: center; margin-top: 1.5rem;
}
.result-title { font-size: 1.7rem; font-weight: 700; margin: 0.4rem 0; }
.result-sub   { font-size: 0.9rem; margin-top: 0.3rem; }
.rec-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem 1.2rem;
    margin-top: 1rem; font-size: 0.9rem; line-height: 1.6;
}
.metric-row { display: flex; gap: 0.8rem; margin-top: 1rem; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 100px;
    background: #f7fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.75rem; text-align: center;
}
.metric-label { font-size: 0.68rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.25rem; font-weight: 700; color: #1a365d; margin-top: 2px; }
.prob-bar-wrap { margin-top: 1rem; }
.prob-label { font-size: 0.8rem; color: #4a5568; margin-bottom: 3px; display: flex; justify-content: space-between; }
.stButton > button {
    background: linear-gradient(135deg, #1a4a7a, #0d2137) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.75rem 2rem !important;
    font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important;
}
.disclaimer { font-size: 0.72rem; color: #a0aec0; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, 'credit_risk_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base, 'credit_risk_meta.json'), 'r') as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()
FEATURE_COLS = meta['feature_cols']
TODAY = pd.Timestamp.today()


def build_features(age, is_male, loan_amount, tenor, monthly_installment,
                   is_topup, contractual_rate, rate, management_fee,
                   monthly_maint_fee, loan_age_months, remaining_months,
                   interest_till_date, loan_value, total_collectable,
                   deductions_till_date, actual_dtd, cumm_maint_fee):

    progress_ratio      = loan_age_months / tenor if tenor > 0 else 0
    interest_to_loan    = interest_till_date / loan_amount if loan_amount > 0 else 0
    loan_value_to_amt   = loan_value / loan_amount if loan_amount > 0 else 0
    total_coll_ratio    = total_collectable / loan_amount if loan_amount > 0 else 0
    monthly_burden      = monthly_installment / loan_amount if loan_amount > 0 else 0
    maint_fee_ratio     = cumm_maint_fee / loan_amount if loan_amount > 0 else 0
    deductions_ratio    = deductions_till_date / total_collectable if total_collectable > 0 else 0
    actual_vs_expected  = actual_dtd / deductions_till_date if deductions_till_date > 0 else 0
    loan_amount_log     = np.log1p(loan_amount)
    loan_value_log      = np.log1p(loan_value)
    installment_log     = np.log1p(monthly_installment)
    total_interest      = interest_till_date

    row = {
        'Age': age, 'IsMale': is_male,
        'LoanAmount': loan_amount, 'LoanAmount_log': loan_amount_log,
        'Tenor': tenor, 'LoanMonthlyInstallment': monthly_installment,
        'Installment_log': installment_log, 'IsTopup': is_topup,
        'ContractualRate': contractual_rate, 'Rate': rate,
        'ManagementFee': management_fee, 'Monthly Maintenance Fee': monthly_maint_fee,
        'LoanAgeMonths': loan_age_months, 'RemainingMonths': remaining_months,
        'ProgressRatio': progress_ratio,
        'InterestToLoan': interest_to_loan, 'LoanValueToAmount': loan_value_to_amt,
        'TotalCollectableRatio': total_coll_ratio, 'MonthlyBurden': monthly_burden,
        'MaintenanceFeeRatio': maint_fee_ratio, 'DeductionsRatio': deductions_ratio,
        'LoanValue': loan_value, 'LoanValue_log': loan_value_log,
        'Total Collectable': total_collectable,
        'InterestTillDate': interest_till_date, 'TotalInterest': total_interest,
        'DeductionsTillDate': deductions_till_date, 'ActualDTD': actual_dtd,
        'ActualVsExpected': actual_vs_expected,
        'Cummulative Maintenance Fee': cumm_maint_fee,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="font-size:2.2rem; margin-bottom:0.4rem;">🏦</div>
  <h1>Credit Risk Early Warning System</h1>
  <p>Nigerian Bank Loan Portfolio · XGBoost · 3-Class Risk Classification<br>
  Trained on 3,509 real loan records · AUC 0.9998 · F1 0.9888</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Loan & Borrower Details")
st.caption("Enter the loan information below to assess credit risk level.")

# ── INPUTS ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("**👤 Borrower**")
    dob = st.date_input("Date of Birth", value=datetime(1985, 1, 1),
                        min_value=datetime(1940, 1, 1), max_value=datetime(2000, 12, 31))
    gender = st.selectbox("Gender", ["Male", "Female"])
    loan_category = st.selectbox("Loan Category", ["NEW LOAN", "TOPUP"])

with col2:
    st.markdown("**💰 Loan Terms**")
    loan_amount = st.number_input("Loan Amount (₦)", min_value=10000, max_value=10000000,
                                   value=300000, step=10000)
    tenor = st.number_input("Tenor (months)", min_value=1, max_value=24, value=12)
    monthly_installment = st.number_input("Monthly Installment (₦)", min_value=1000,
                                          max_value=2000000, value=40000, step=1000)

st.markdown("")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**📊 Rates & Fees**")
    contractual_rate = st.number_input("Contractual Rate (%)", min_value=0.0,
                                       max_value=30.0, value=4.75, step=0.1)
    rate = st.number_input("Effective Rate (%)", min_value=0.0,
                           max_value=30.0, value=7.73, step=0.1)
    management_fee = st.number_input("Management Fee (₦)", min_value=0,
                                     max_value=100000, value=4250, step=250)
    monthly_maint_fee = st.number_input("Monthly Maintenance Fee (₦)", min_value=0,
                                        max_value=10000, value=785, step=50)

with col4:
    st.markdown("**📅 Portfolio Position**")
    loan_age_months = st.number_input("Months Since Disbursement", min_value=0,
                                      max_value=24, value=3)
    remaining_months = st.number_input("Months Remaining to Maturity", min_value=0,
                                       max_value=24, value=9)
    deductions_till_date = st.number_input("Total Deductions to Date (₦)", min_value=0,
                                           max_value=5000000, value=120000, step=5000)
    actual_dtd = st.number_input("Actual Deductions to Date (₦)", min_value=0,
                                 max_value=5000000, value=120000, step=5000)

st.markdown("")
col5, col6 = st.columns(2)

with col5:
    st.markdown("**🏦 Financials**")
    loan_value = st.number_input("Loan Value (₦)", min_value=10000,
                                 max_value=10000000, value=380000, step=10000)
    total_collectable = st.number_input("Total Collectable (₦)", min_value=10000,
                                        max_value=10000000, value=488000, step=10000)
    interest_till_date = st.number_input("Interest Accrued to Date (₦)", min_value=0,
                                         max_value=2000000, value=45000, step=1000)

with col6:
    st.markdown("**🔢 Cumulative**")
    cumm_maint_fee = st.number_input("Cumulative Maintenance Fee (₦)", min_value=0,
                                     max_value=500000, value=2355, step=100)

st.markdown("")

# ── PREDICT ──────────────────────────────────────────────
if st.button("🔍 Assess Credit Risk"):

    age     = int((pd.Timestamp.today() - pd.Timestamp(dob)).days / 365.25)
    is_male = 1 if gender == "Male" else 0
    is_topup= 1 if loan_category == "TOPUP" else 0

    try:
        X_input = build_features(
            age, is_male, loan_amount, tenor, monthly_installment,
            is_topup, contractual_rate, rate, management_fee,
            monthly_maint_fee, loan_age_months, remaining_months,
            interest_till_date, loan_value, total_collectable,
            deductions_till_date, actual_dtd, cumm_maint_fee
        )
        X_input = X_input.replace([np.inf, -np.inf], np.nan).fillna(0)

        probs     = model.predict_proba(X_input)[0]
        pred_class= int(np.argmax(probs))
        prob_low, prob_med, prob_high = probs[0], probs[1], probs[2]

        # ── Result card ──────────────────────────────────
        if pred_class == 0:
            st.markdown(f"""
            <div class="result-low">
              <div style="font-size:2.5rem">🟢</div>
              <div class="result-title" style="color:#276749">LOW RISK</div>
              <div class="result-sub" style="color:#276749">Loan is performing within acceptable parameters</div>
              <div class="rec-box" style="text-align:left">
                <strong>Recommendation:</strong> Loan is performing satisfactorily.
                Continue standard monitoring cadence. No immediate action required.
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif pred_class == 1:
            st.markdown(f"""
            <div class="result-medium">
              <div style="font-size:2.5rem">🟡</div>
              <div class="result-title" style="color:#92400e">MEDIUM RISK</div>
              <div class="result-sub" style="color:#92400e">Early warning signals detected — Watchlist category</div>
              <div class="rec-box" style="text-align:left">
                <strong>Recommendation:</strong> Escalate to relationship manager for review.
                Increase monitoring frequency. Consider proactive engagement with borrower
                to restructure if delinquency trends continue.
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-high">
              <div style="font-size:2.5rem">🔴</div>
              <div class="result-title" style="color:#c53030">HIGH RISK</div>
              <div class="result-sub" style="color:#c53030">Significant credit deterioration — Substandard / Loss category</div>
              <div class="rec-box" style="text-align:left">
                <strong>Recommendation:</strong> Immediate escalation to credit risk committee required.
                Initiate recovery proceedings. Assess collateral adequacy and consider
                provisioning in line with CBN prudential guidelines.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability bars ──────────────────────────────
        st.markdown("")
        st.markdown("#### Risk Probability Breakdown")
        st.markdown(f"""
        <div class="prob-bar-wrap">
          <div class="prob-label"><span>🟢 Low Risk (Performing)</span><span><b>{prob_low:.1%}</b></span></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob_low))

        st.markdown(f"""
        <div class="prob-bar-wrap">
          <div class="prob-label"><span>🟡 Medium Risk (Watchlist)</span><span><b>{prob_med:.1%}</b></span></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob_med))

        st.markdown(f"""
        <div class="prob-bar-wrap">
          <div class="prob-label"><span>🔴 High Risk (Substandard / Loss)</span><span><b>{prob_high:.1%}</b></span></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob_high))

        # ── Key computed metrics ──────────────────────────
        progress_pct = (loan_age_months / tenor * 100) if tenor > 0 else 0
        repay_ratio  = deductions_till_date / total_collectable if total_collectable > 0 else 0

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="metric-label">Borrower Age</div>
            <div class="metric-value">{age} yrs</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Loan Progress</div>
            <div class="metric-value">{progress_pct:.0f}%</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Repayment Ratio</div>
            <div class="metric-value">{repay_ratio:.1%}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Monthly Burden</div>
            <div class="metric-value">{monthly_installment/loan_amount*100:.1f}%</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Effective Rate</div>
            <div class="metric-value">{rate:.2f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── ABOUT ─────────────────────────────────────────────────
with st.expander("ℹ️ About this system"):
    st.markdown(f"""
    **Credit Risk Early Warning System** — built as a Machine Learning class project.

    **Dataset:** Nigerian bank loan portfolio (Ogun State), 3,509 loan records.

    **Model:** XGBoost — 3-class ordinal risk classifier.

    **Risk levels mirror CBN loan classification guidelines:**
    - 🟢 **Low Risk** → Performing loans
    - 🟡 **Medium Risk** → Watchlist 1, 2, 3 (early deterioration signals)
    - 🔴 **High Risk** → Substandard, Loss, Doubtful

    **Performance (test set, 702 loans):**
    - F1 Macro: **{meta['test_f1_macro']}**
    - AUC (One-vs-Rest): **{meta['test_auc_ovr']}**
    - Accuracy: **99%**

    **Why XGBoost?** Outperformed Logistic Regression and Random Forest on all metrics.
    Class imbalance (2414 / 875 / 220) handled with SMOTE oversampling.

    **Leakage prevention:** PDOs, VAR, Months in Delinquent, and Amount in Delinquent
    were excluded from model features as they are derived from the classification labels.

    ⚠️ For educational purposes. Based on real company data used with permission.
    """)

st.markdown("""
<div class="disclaimer">
  Credit Risk Early Warning System · Nigerian Bank Loan Portfolio · XGBoost + SMOTE<br>
  Built for academic purposes · Not for production lending decisions
</div>
""", unsafe_allow_html=True)
