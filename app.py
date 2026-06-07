import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="CreditSense - Loan Default Prediction",
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
.cbn-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem;
}
.cbn-low    { background: #c6f6d5; color: #276749; }
.cbn-medium { background: #fef3c7; color: #92400e; }
.cbn-high   { background: #fed7d7; color: #c53030; }
.rec-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem 1.2rem;
    margin-top: 1rem; font-size: 0.9rem; line-height: 1.6; text-align: left;
}
.metric-row { display: flex; gap: 0.8rem; margin-top: 1rem; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 100px;
    background: #f7fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.75rem; text-align: center;
}
.metric-label { font-size: 0.68rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.25rem; font-weight: 700; color: #1a365d; margin-top: 2px; }
.prob-label { font-size: 0.8rem; color: #4a5568; margin-bottom: 3px; display: flex; justify-content: space-between; }
.batch-info {
    background: #EBF4FF; border: 1px solid #BEE3F8;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.88rem; line-height: 1.7; margin-bottom: 1rem;
}
.stButton > button {
    background: linear-gradient(135deg, #1a4a7a, #0d2137) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.75rem 2rem !important;
    font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important;
}
.tab-note { font-size: 0.82rem; color: #718096; margin-bottom: 1rem; }
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

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
CBN_LABELS  = {0: "Performing", 1: "Watchlist", 2: "Substandard / Loss / Doubtful"}


def engineer_row(age, is_male, loan_amount, tenor, monthly_installment,
                 is_topup, contractual_rate, rate, management_fee,
                 monthly_maint_fee, creation_month, creation_dow, creation_quarter):

    monthly_burden       = monthly_installment / loan_amount if loan_amount > 0 else 0
    eff_spread           = rate - contractual_rate
    mgmt_fee_ratio       = management_fee / loan_amount if loan_amount > 0 else 0
    maint_fee_ratio      = monthly_maint_fee / loan_amount if loan_amount > 0 else 0
    total_cost_ratio     = (management_fee + (monthly_maint_fee * tenor)) / loan_amount if loan_amount > 0 else 0
    loan_per_month       = loan_amount / tenor if tenor > 0 else 0

    return {
        'Age': age, 'IsMale': is_male,
        'LoanAmount': loan_amount, 'LoanAmount_log': np.log1p(loan_amount),
        'Tenor': tenor, 'LoanMonthlyInstallment': monthly_installment,
        'Installment_log': np.log1p(monthly_installment), 'IsTopup': is_topup,
        'ContractualRate': contractual_rate, 'Rate': rate,
        'EffectiveSpread': eff_spread,
        'ManagementFee': management_fee, 'Monthly Maintenance Fee': monthly_maint_fee,
        'MonthlyBurden': monthly_burden, 'ManagementFeeRatio': mgmt_fee_ratio,
        'MaintenanceFeeRatio': maint_fee_ratio, 'TotalCostRatio': total_cost_ratio,
        'LoanPerMonth': loan_per_month,
        'CreationMonth': creation_month, 'CreationDayOfWeek': creation_dow, 'CreationQuarter': creation_quarter,
    }


def batch_predict(df_raw):
    df = df_raw.copy()
    df['DateofBirth'] = pd.to_datetime(df['DateofBirth'], errors='coerce')
    df['Age'] = ((TODAY - df['DateofBirth']).dt.days / 365.25).fillna(35).astype(int)
    df['IsMale']  = (df['Gender'].astype(str).str.strip().str.lower() == 'male').astype(int)
    df['IsTopup'] = (df['LoanCategory'].astype(str).str.strip().str.upper() == 'TOPUP').astype(int)
    df['CreationDate'] = pd.to_datetime(df.get('CreationDate', TODAY), errors='coerce')

    def col(name): return pd.to_numeric(df.get(name, 0), errors='coerce').fillna(0)

    rows = []
    for i in range(len(df)):
        cd = df['CreationDate'].iloc[i] if pd.notnull(df['CreationDate'].iloc[i]) else TODAY
        rows.append(engineer_row(
            age=int(df['Age'].iloc[i]),
            is_male=int(df['IsMale'].iloc[i]),
            loan_amount=col('LoanAmount').iloc[i],
            tenor=col('Tenor').iloc[i],
            monthly_installment=col('LoanMonthlyInstallment').iloc[i],
            is_topup=int(df['IsTopup'].iloc[i]),
            contractual_rate=col('ContractualRate').iloc[i],
            rate=col('Rate').iloc[i],
            management_fee=col('ManagementFee').iloc[i],
            monthly_maint_fee=col('Monthly Maintenance Fee').iloc[i],
            creation_month=cd.month, creation_dow=cd.dayofweek, creation_quarter=cd.quarter,
        ))

    X = pd.DataFrame(rows)[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    result = df_raw.copy()
    result['Predicted Risk Level']    = [RISK_LABELS[p] for p in preds]
    result['Predicted CBN Class']     = [CBN_LABELS[p] for p in preds]
    result['Prob Low Risk (%)']       = (probs[:, 0] * 100).round(1)
    result['Prob Medium Risk (%)']    = (probs[:, 1] * 100).round(1)
    result['Prob High Risk (%)']      = (probs[:, 2] * 100).round(1)
    return result, preds, probs


def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Default Predictions')
    return output.getvalue()


st.markdown("""
<div class="hero">
  <div style="font-size:2.2rem; margin-bottom:0.4rem;">🏦</div>
  <h1>CreditSense</h1>
  <p>Loan Default Prediction at Origination for Nigerian Consumer Lending<br>
  XGBoost · CBN Risk Rating · Trained on 3,509 Real Loan Records<br>
  Test AUC 0.9582 · F1 Macro 0.7668 · Accuracy 87%</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Loan Prediction", "Portfolio Batch Prediction"])

# TAB 1 - SINGLE LOAN
with tab1:
    st.markdown('<div class="tab-note">Enter loan application details to predict the likely risk outcome based on data available at origination.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Borrower**")
        dob = st.date_input("Date of Birth", value=datetime(1985, 1, 1),
                            min_value=datetime(1940, 1, 1), max_value=datetime(2007, 12, 31))
        gender = st.selectbox("Gender", ["Male", "Female"])
        loan_category = st.selectbox("Loan Category", ["NEW LOAN", "TOPUP"])

    with col2:
        st.markdown("**Loan Terms**")
        loan_amount = st.number_input("Loan Amount (N)", min_value=10000, max_value=10000000, value=300000, step=10000)
        tenor = st.number_input("Tenor (months)", min_value=1, max_value=24, value=12)
        monthly_installment = st.number_input("Monthly Installment (N)", min_value=1000, max_value=2000000, value=40000, step=1000)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Rates**")
        contractual_rate = st.number_input("Contractual Rate (%)", min_value=0.0, max_value=30.0, value=4.75, step=0.1)
        rate = st.number_input("Effective Rate (%)", min_value=0.0, max_value=30.0, value=7.73, step=0.1)

    with col4:
        st.markdown("**Fees**")
        management_fee = st.number_input("Management Fee (N)", min_value=0, max_value=100000, value=4250, step=250)
        monthly_maint_fee = st.number_input("Monthly Maintenance Fee (N)", min_value=0, max_value=10000, value=785, step=50)

    st.markdown("")
    if st.button("Predict Default Risk", key="single"):
        age     = int((TODAY - pd.Timestamp(dob)).days / 365.25)
        is_male = 1 if gender == "Male" else 0
        is_topup = 1 if loan_category == "TOPUP" else 0
        now = pd.Timestamp.today()

        try:
            row = engineer_row(age, is_male, loan_amount, tenor, monthly_installment,
                               is_topup, contractual_rate, rate, management_fee,
                               monthly_maint_fee, now.month, now.dayofweek, now.quarter)
            X_input = pd.DataFrame([row])[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
            probs      = model.predict_proba(X_input)[0]
            pred_class = int(np.argmax(probs))
            prob_low, prob_med, prob_high = probs[0], probs[1], probs[2]

            if pred_class == 0:
                st.markdown(f"""
                <div class="result-low">
                  <div style="font-size:2.5rem">🟢</div>
                  <div class="result-title" style="color:#276749">LIKELY TO PERFORM</div>
                  <div class="cbn-badge cbn-low">Predicted CBN Rating: Performing</div>
                  <div class="result-sub" style="color:#276749; margin-top:0.5rem;">Low predicted probability of default</div>
                  <div class="rec-box"><strong>Recommendation:</strong> Loan profile predicts performance within acceptable parameters. Approve under standard terms with routine monitoring.</div>
                </div>""", unsafe_allow_html=True)

            elif pred_class == 1:
                st.markdown(f"""
                <div class="result-medium">
                  <div style="font-size:2.5rem">🟡</div>
                  <div class="result-title" style="color:#92400e">LIKELY WATCHLIST</div>
                  <div class="cbn-badge cbn-medium">Predicted CBN Rating: Watchlist</div>
                  <div class="result-sub" style="color:#92400e; margin-top:0.5rem;">Moderate predicted probability of default</div>
                  <div class="rec-box"><strong>Recommendation:</strong> Loan profile suggests elevated risk of falling into Watchlist categories. Consider tightening loan terms (lower amount, shorter tenor, additional guarantees) or refer to credit committee for review.</div>
                </div>""", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-high">
                  <div style="font-size:2.5rem">🔴</div>
                  <div class="result-title" style="color:#c53030">LIKELY DEFAULT</div>
                  <div class="cbn-badge cbn-high">Predicted CBN Rating: Substandard / Loss / Doubtful</div>
                  <div class="result-sub" style="color:#c53030; margin-top:0.5rem;">High predicted probability of default</div>
                  <div class="rec-box"><strong>Recommendation:</strong> Loan profile shows high predicted risk of severe deterioration. Decline application, or refer to credit risk committee for detailed review and possible restructuring of terms before approval.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("#### Predicted Risk Probability")
            st.markdown(f'<div class="prob-label"><span>🟢 Will Perform</span><span><b>{prob_low:.1%}</b></span></div>', unsafe_allow_html=True)
            st.progress(float(prob_low))
            st.markdown(f'<div class="prob-label"><span>🟡 Will be Watchlisted</span><span><b>{prob_med:.1%}</b></span></div>', unsafe_allow_html=True)
            st.progress(float(prob_med))
            st.markdown(f'<div class="prob-label"><span>🔴 Will Default</span><span><b>{prob_high:.1%}</b></span></div>', unsafe_allow_html=True)
            st.progress(float(prob_high))

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box"><div class="metric-label">Borrower Age</div><div class="metric-value">{age} yrs</div></div>
              <div class="metric-box"><div class="metric-label">Monthly Burden</div><div class="metric-value">{monthly_installment/loan_amount*100:.1f}%</div></div>
              <div class="metric-box"><div class="metric-label">Total Cost</div><div class="metric-value">{((management_fee + monthly_maint_fee*tenor)/loan_amount*100):.1f}%</div></div>
              <div class="metric-box"><div class="metric-label">Effective Rate</div><div class="metric-value">{rate:.2f}%</div></div>
              <div class="metric-box"><div class="metric-label">Loan Type</div><div class="metric-value">{loan_category}</div></div>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# TAB 2 - BATCH
with tab2:
    st.markdown('<div class="tab-note">Upload a portfolio file of new loan applications to predict default risk for all at once.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="batch-info">
      <strong>How it works:</strong><br>
      1. Upload your portfolio file as Excel (.xlsx) or CSV<br>
      2. The model predicts default risk for every loan based on origination features<br>
      3. Download the results as Excel with predictions for all loans<br><br>
      <strong>Required columns:</strong> DateofBirth, Gender, LoanCategory, LoanAmount, Tenor,
      LoanMonthlyInstallment, ContractualRate, Rate, ManagementFee, Monthly Maintenance Fee, CreationDate
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload portfolio file", type=["xlsx", "csv"])

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.success(f"File loaded: {len(df_upload):,} loans detected.")

            if st.button("Predict for Entire Portfolio", key="batch"):
                with st.spinner("Running default predictions..."):
                    result_df, preds, probs = batch_predict(df_upload)

                low_count  = int((preds == 0).sum())
                med_count  = int((preds == 1).sum())
                high_count = int((preds == 2).sum())
                total      = len(preds)

                st.markdown("#### Portfolio Default Prediction Summary")
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-box">
                    <div class="metric-label">Total Loans</div>
                    <div class="metric-value">{total:,}</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">🟢 Will Perform</div>
                    <div class="metric-value" style="color:#276749">{low_count:,} ({low_count/total:.1%})</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">🟡 Watchlist Predicted</div>
                    <div class="metric-value" style="color:#92400e">{med_count:,} ({med_count/total:.1%})</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">🔴 Default Predicted</div>
                    <div class="metric-value" style="color:#c53030">{high_count:,} ({high_count/total:.1%})</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                st.markdown("#### High-Risk Applications")
                flagged = result_df[result_df['Predicted Risk Level'].isin(['Medium Risk', 'High Risk'])].copy()
                display_cols = [c for c in ['ContractId', 'CustomerId', 'LoanAmount', 'Tenor',
                                            'Predicted Risk Level', 'Predicted CBN Class',
                                            'Prob Medium Risk (%)', 'Prob High Risk (%)'] if c in flagged.columns]
                st.dataframe(flagged[display_cols].reset_index(drop=True), use_container_width=True)

                st.markdown("")
                excel_data = to_excel(result_df)
                st.download_button(
                    label="Download Prediction Report (Excel)",
                    data=excel_data,
                    file_name=f"CreditSense_Predictions_{datetime.today().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your file has the required columns listed above.")


with st.expander("About this system"):
    st.markdown(f"""
    **CreditSense** is a machine learning powered loan default prediction system developed
    as part of an MSc Machine Learning course project.

    **What it does:** Predicts the likely CBN risk rating of a loan based on data available
    at the point of loan origination (application time), before the loan is disbursed.

    **Dataset:** Anonymized Nigerian consumer bank loan portfolio, 3,509 loan records.
Data sourced from a licensed Nigerian financial institution with permission.

    **Model:** XGBoost classifier trained on 21 origination-time features only.
    Portfolio position features (deductions, days past due, etc.) were deliberately excluded
    to ensure the model is genuinely predictive rather than descriptive.

    Class imbalance addressed using SMOTE (Synthetic Minority Oversampling Technique).

    **CBN Risk Rating Framework:**
    - Performing: loans expected to repay normally
    - Watchlist 1 / 2 / 3: loans likely to show early warning signs (1-90 days past due)
    - Substandard / Doubtful / Loss: loans likely to default

    **Model Performance (held-out test set, 702 loans):**
    - F1 Macro: **{meta['test_f1_macro']}**
    - F1 Weighted: **{meta['test_f1_weighted']}**
    - AUC (One vs Rest): **{meta['test_auc_ovr']}**
    - Overall Accuracy: **87%**

    The model achieves these results using only features known at loan origination,
    making it suitable for use in credit underwriting decisions.

    This tool is intended for use by trained credit officers as a decision support system.
    It does not replace professional credit judgment or CBN regulatory requirements.
    """)

st.markdown("""
<div class="disclaimer">
  CreditSense · Nigerian Consumer Lending Default Prediction · XGBoost + SMOTE<br>
  MSc Machine Learning Project · For academic and internal review purposes only
</div>
""", unsafe_allow_html=True)
