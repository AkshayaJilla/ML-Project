import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main website background */
    .stApp {
        background: linear-gradient(
            135deg,
            #eaf4ff 0%,
            #f8fbff 45%,
            #e8f5f0 100%
        );
    }

    /* Main headings */
    h1, h2, h3 {
        color: #164e78 !important;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #123b63;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .sub-title {
        text-align: center;
        color: #526779;
        font-size: 18px;
        margin-bottom: 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #123b63,
            #176b87
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Input area */
    .input-box {
        background-color: #ffffff;
        border: 1px solid #c7dceb;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }

    /* Approved result */
    .approved-heading {
        color: #087443 !important;
        font-size: 38px !important;
        font-weight: 900 !important;
        text-align: center;
        margin-bottom: 5px;
    }

    .approved-description {
        color: #276749;
        font-size: 16px;
        text-align: center;
    }

    /* Rejected result */
    .rejected-heading {
        color: #b42318 !important;
        font-size: 38px !important;
        font-weight: 900 !important;
        text-align: center;
        margin-bottom: 5px;
    }

    .rejected-description {
        color: #9b2c2c;
        font-size: 16px;
        text-align: center;
    }

    /* Compact repayment section */
    .repayment-heading {
        color: #164e78 !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .repayment-line {
        font-size: 17px;
        color: #243b53;
        padding: 2px 0;
        margin: 0;
    }

    .repayment-value {
        font-size: 19px;
        font-weight: 700;
        color: #164e78;
        padding: 0;
        margin: 0 0 5px 0;
    }

    /* EMI highlight */
    .emi-heading {
        color: #087443 !important;
        font-size: 25px !important;
        font-weight: 800 !important;
        margin-top: 8px;
        margin-bottom: 2px;
    }

    .emi-value {
        color: #087443 !important;
        font-size: 35px !important;
        font-weight: 900 !important;
        margin-top: 0;
        margin-bottom: 2px;
    }

    .emi-description {
        color: #526779;
        font-size: 15px;
        margin-top: 0;
    }

    /* Information box */
    .info-box {
        background-color: #e2f0fb;
        border-left: 5px solid #2477b5;
        border-radius: 8px;
        padding: 12px;
        color: #164e78;
        font-size: 16px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #526779;
        font-size: 14px;
        padding: 20px;
        margin-top: 20px;
    }

    /* Button color */
    div.stButton > button {
        background-color: #176b87;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 17px;
        font-weight: 700;
        padding: 12px;
    }

    div.stButton > button:hover {
        background-color: #0e4f68;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "loan_approval_model.pkl"
DATA_PATH = BASE_DIR / "loan_applications.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "loan_approval_model.pkl was not found. "
            "Please keep loan_approval_model.pkl in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return None


# ============================================================
# EMI CALCULATION FUNCTION
# ============================================================

def calculate_emi(principal, annual_rate, months):
    """
    Calculate monthly EMI, total payable amount,
    and total interest.
    """

    monthly_rate = annual_rate / (12 * 100)

    if monthly_rate == 0:
        emi = principal / months

    else:
        emi = (
            principal
            * monthly_rate
            * (1 + monthly_rate) ** months
        ) / (
            (1 + monthly_rate) ** months - 1
        )

    total_payment = emi * months
    total_interest = total_payment - principal

    return emi, total_payment, total_interest


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:
    model = load_model()
    df = load_data()

except Exception as error:
    st.error("Unable to load the project files.")
    st.code(str(error))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-powered loan approval prediction using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write(
        "Enter the applicant's personal and financial details "
        "to predict whether the loan application may be approved."
    )

    st.divider()

    st.write("🤖 **Algorithm:** Random Forest Classifier")
    st.write("📊 **Dataset:** 2,000 applications")
    st.write("🧠 **Task:** Binary Classification")
    st.write("💰 **Feature:** Monthly EMI Calculator")

    st.divider()

    st.info(
        "The trained model supports applicants between "
        "21 and 60 years of age."
    )


# ============================================================
# APPLICANT INPUT SECTION
# ============================================================

st.markdown(
    '<div class="input-box">',
    unsafe_allow_html=True
)

st.subheader("📝 Applicant Details")

left_column, right_column = st.columns(2)


with left_column:

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=60,
        value=30,
        step=1
    )

    employment_status = st.selectbox(
        "Employment Status",
        [
            "Employed",
            "Self-Employed",
            "Unemployed"
        ]
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=6,
        value=1,
        step=1
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semi-Urban",
            "Rural"
        ]
    )


with right_column:

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=7000.0,
        max_value=1000000.0,
        value=30000.0,
        step=1000.0
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=20000.0,
        max_value=5000000.0,
        value=150000.0,
        step=5000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60],
        index=2
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EMI SETTINGS
# ============================================================

st.subheader("💰 EMI Calculation Settings")

interest_rate = st.number_input(
    "Annual Interest Rate (%)",
    min_value=0.0,
    max_value=30.0,
    value=10.0,
    step=0.5
)

st.caption(
    "The interest rate is used only to calculate the estimated EMI. "
    "It is not used by the machine learning model."
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Loan Approval",
    use_container_width=True
)


# ============================================================
# PREDICTION PROCESS
# ============================================================

if predict_button:

    # The current trained model expects Gender.
    # Gender is kept as Male to match the existing trained model.
    input_data = pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Age": age,
                "MonthlyIncome": monthly_income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "EmploymentStatus": employment_status,
                "ExistingLoans": existing_loans,
                "LoanTerm": loan_term,
                "PropertyArea": property_area
            }
        ]
    )

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = int(model.predict(input_data)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(input_data)[0][1]
            )


        # ----------------------------------------------------
        # APPROVED OR REJECTED RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.markdown(
                '<div class="approved-heading">'
                '✅ LOAN APPROVED'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="approved-description">'
                'The model predicts that this application is likely to be approved.'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="rejected-heading">'
                '❌ LOAN REJECTED'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="rejected-description">'
                'The model predicts that this application is likely to be rejected.'
                '</div>',
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # APPROVAL PROBABILITY
        # ----------------------------------------------------

        if probability is not None:

            st.subheader("📊 Approval Probability")

            st.progress(probability)

            st.metric(
                "Estimated Approval Probability",
                f"{probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # EMI CALCULATION
        # ----------------------------------------------------

        if prediction == 1:

            emi, total_payment, total_interest = calculate_emi(
                loan_amount,
                interest_rate,
                loan_term
            )

            st.markdown(
                '<div class="repayment-heading">'
                '💰 Loan Repayment Details'
                '</div>',
                unsafe_allow_html=True
            )

            # Monthly EMI
            st.markdown(
                '<div class="emi-heading">Monthly EMI</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="emi-value">₹{emi:,.2f}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="emi-description">'
                f'Estimated amount to pay every month for {loan_term} months.'
                '</div>',
                unsafe_allow_html=True
            )

            st.divider()

            # Compact listed repayment details
            st.markdown(
                '<div class="repayment-line"><b>Loan Amount</b></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="repayment-value">₹{loan_amount:,.2f}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="repayment-line"><b>Loan Term</b></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="repayment-value">{loan_term} Months</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="repayment-line"><b>Annual Interest Rate</b></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="repayment-value">{interest_rate:.1f}%</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="repayment-line"><b>Total Amount Payable</b></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="repayment-value">₹{total_payment:,.2f}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="repayment-line"><b>Total Interest</b></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="repayment-value">₹{total_interest:,.2f}</div>',
                unsafe_allow_html=True
            )

            st.info(
                f"You would pay approximately ₹{emi:,.2f} "
                f"every month for {loan_term} months."
            )


        # ----------------------------------------------------
        # APPLICATION SUMMARY
        # ----------------------------------------------------

        st.subheader("📋 Application Summary")

        summary = pd.DataFrame(
            {
                "Field": [
                    "Age",
                    "Monthly Income",
                    "Loan Amount",
                    "Credit Score",
                    "Employment Status",
                    "Existing Loans",
                    "Loan Term",
                    "Property Area",
                    "Annual Interest Rate"
                ],
                "Value": [
                    f"{int(age)} years",
                    f"₹{monthly_income:,.0f}",
                    f"₹{loan_amount:,.0f}",
                    str(int(credit_score)),
                    employment_status,
                    str(int(existing_loans)),
                    f"{int(loan_term)} months",
                    property_area,
                    f"{interest_rate:.1f}%"
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error("Prediction failed. Please check the input values.")
        st.code(str(error))


# ============================================================
# DATASET DASHBOARD
# ============================================================

if df is not None:

    st.divider()

    st.subheader("📊 Dataset Overview")

    total_applications = len(df)

    approved_applications = int(
        df["LoanApproved"].sum()
    )

    rejected_applications = (
        total_applications - approved_applications
    )

    approval_rate = (
        approved_applications / total_applications
    ) * 100


    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Total Applications",
        f"{total_applications:,}"
    )

    metric_2.metric(
        "Approved",
        f"{approved_applications:,}"
    )

    metric_3.metric(
        "Rejected",
        f"{rejected_applications:,}"
    )

    metric_4.metric(
        "Approval Rate",
        f"{approval_rate:.1f}%"
    )


    st.subheader("📈 Approval Distribution")

    chart_data = pd.DataFrame(
        {
            "Status": [
                "Approved",
                "Rejected"
            ],
            "Applications": [
                approved_applications,
                rejected_applications
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Status")
    )


    with st.expander("🔍 View Sample Dataset"):

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>Loan Approval Prediction System</b><br>
        Machine Learning Project • Random Forest • Streamlit<br>
        Monthly EMI Calculator Included
    </div>
    """,
    unsafe_allow_html=True
)
