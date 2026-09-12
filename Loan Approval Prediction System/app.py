import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page background */
    .stApp {
        background-color: #f4f7fb;
        color: #1f2937;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #123c69 0%,
            #1d5d91 50%,
            #2784b5 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main title */
    .main-title {
        color: #123c69;
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #52606d;
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-heading {
        color: #123c69;
        font-size: 27px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 3px solid #2b7bbb;
        padding-bottom: 8px;
    }

    /* Input card */
    .input-card {
        background-color: white;
        border: 1px solid #d8e2ef;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(18, 60, 105, 0.08);
    }

    /* Prediction approved card */
    .approved-card {
        background: linear-gradient(
            135deg,
            #d9fbe5,
            #b8f1cd
        );
        border: 2px solid #1f9d55;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .approved-title {
        color: #126b35;
        font-size: 32px;
        font-weight: 800;
    }

    .approved-text {
        color: #155d35;
        font-size: 18px;
    }

    /* Prediction rejected card */
    .rejected-card {
        background: linear-gradient(
            135deg,
            #ffe1e1,
            #ffc4c4
        );
        border: 2px solid #d62828;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .rejected-title {
        color: #a4161a;
        font-size: 32px;
        font-weight: 800;
    }

    .rejected-text {
        color: #8a1518;
        font-size: 18px;
    }

    /* Probability card */
    .probability-card {
        background-color: #e5f1ff;
        border-left: 6px solid #1769aa;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .probability-title {
        color: #123c69;
        font-size: 22px;
        font-weight: 750;
    }

    /* EMI card */
    .emi-card {
        background: linear-gradient(
            135deg,
            #fff4cc,
            #ffe69a
        );
        border: 2px solid #d99a00;
        border-radius: 16px;
        padding: 22px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .emi-title {
        color: #805b00;
        font-size: 26px;
        font-weight: 800;
    }

    .emi-amount {
        color: #9a6700;
        font-size: 32px;
        font-weight: 800;
    }

    .emi-description {
        color: #795600;
        font-size: 16px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #52606d;
        font-size: 14px;
        padding: 25px;
        margin-top: 35px;
        border-top: 1px solid #d8e2ef;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1769aa;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-size: 17px;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #123c69;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "loan_approval_model.pkl"
DATA_PATH = BASE_DIR / "loan_applications.csv"


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Please upload "
            "'loan_approval_model.pkl' to the GitHub folder."
        )
        st.stop()

    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <h1 style="text-align:center;">🏦</h1>
        <h2 style="text-align:center;">Loan Approval</h2>
        <h2 style="text-align:center;">Prediction System</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.write("### 📌 Project Information")
    st.write(
        """
        This project uses Machine Learning to predict
        whether a loan application may be approved
        or rejected.
        """
    )

    st.write("### 🤖 Machine Learning Model")
    st.write("Random Forest Classifier")

    st.write("### 📊 Dataset")
    st.write("2,000 loan applications")

    st.write("### 🎯 Problem Type")
    st.write("Binary Classification")

    st.write("### 💻 Technologies")
    st.write("Python, Pandas, Scikit-learn and Streamlit")

    st.markdown("---")

    st.info(
        "Enter the applicant details and click "
        "'Predict Loan Approval' to get the result."
    )


# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Enter the applicant's details to predict whether the loan
    application is likely to be approved or rejected.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# APPLICANT DETAILS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">👤 Applicant Details</div>',
    unsafe_allow_html=True
)

with st.container():

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=21,
            max_value=60,
            value=30,
            step=1
        )

    with col2:
        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=7000.0,
            max_value=1000000.0,
            value=40000.0,
            step=1000.0
        )

    with col3:
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=20000.0,
            max_value=5000000.0,
            value=150000.0,
            step=5000.0
        )


# ---------------------------------------------------------
# FINANCIAL DETAILS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">💰 Financial Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700,
        step=1
    )

with col2:
    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=6,
        value=1,
        step=1
    )

with col3:
    loan_term = st.selectbox(
        "Loan Term",
        options=[12, 24, 36, 48, 60],
        index=2,
        format_func=lambda x: f"{x} Months"
    )


# ---------------------------------------------------------
# EMPLOYMENT AND PROPERTY DETAILS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">🏠 Employment and Property Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    employment_status = st.selectbox(
        "Employment Status",
        options=[
            "Employed",
            "Self-Employed",
            "Unemployed"
        ]
    )

with col2:
    property_area = st.selectbox(
        "Property Area",
        options=[
            "Urban",
            "Semi-Urban",
            "Rural"
        ]
    )


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Loan Approval"
)


# ---------------------------------------------------------
# PREDICTION AND RESULTS
# ---------------------------------------------------------

if predict_button:

    try:

        # The current trained model includes Gender.
        # Since the original input form does not ask Gender,
        # Male is used as the default value.
        input_data = pd.DataFrame(
            {
                "Gender": ["Male"],
                "Age": [age],
                "MonthlyIncome": [monthly_income],
                "LoanAmount": [loan_amount],
                "CreditScore": [credit_score],
                "EmploymentStatus": [employment_status],
                "ExistingLoans": [existing_loans],
                "LoanTerm": [loan_term],
                "PropertyArea": [property_area]
            }
        )

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability_values = model.predict_proba(input_data)[0]
            approval_probability = probability_values[1] * 100
        else:
            approval_probability = 100 if prediction == 1 else 0

        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.markdown(
                """
                <div class="approved-card">
                    <div class="approved-title">
                        ✅ LOAN APPROVED
                    </div>
                    <div class="approved-text">
                        The model predicts that this application
                        is likely to be approved.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="rejected-card">
                    <div class="rejected-title">
                        ❌ LOAN REJECTED
                    </div>
                    <div class="rejected-text">
                        The model predicts that this application
                        may not be approved.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # APPROVAL PROBABILITY
        # -------------------------------------------------

        st.markdown(
            """
            <div class="probability-card">
                <div class="probability-title">
                    📊 Approval Probability
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            int(round(approval_probability))
        )

        st.metric(
            label="Estimated Approval Probability",
            value=f"{approval_probability:.2f}%"
        )


        # -------------------------------------------------
        # LOAN REPAYMENT DETAILS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-heading">💰 Loan Repayment Details</div>',
            unsafe_allow_html=True
        )

        annual_interest_rate = 10.0

        monthly_interest_rate = (
            annual_interest_rate / 12 / 100
        )

        number_of_months = loan_term

        principal = loan_amount

        if monthly_interest_rate == 0:

            monthly_emi = principal / number_of_months

        else:

            monthly_emi = (
                principal
                * monthly_interest_rate
                * (1 + monthly_interest_rate) ** number_of_months
                / (
                    (1 + monthly_interest_rate) ** number_of_months - 1
                )
            )

        total_amount_payable = monthly_emi * number_of_months

        total_interest = total_amount_payable - principal


        # -------------------------------------------------
        # EMI DETAILS
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="emi-card">
                <div class="emi-title">
                    💰 Monthly EMI
                </div>

                <div class="emi-amount">
                    ₹{monthly_emi:,.2f}
                </div>

                <div class="emi-description">
                    Estimated amount to pay every month
                    for {loan_term} months.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # REPAYMENT INFORMATION IN LISTED ORDER
        # -------------------------------------------------

        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 14px;
                padding: 20px;
                border: 1px solid #d8e2ef;
                margin-bottom: 15px;
            ">
                <h2 style="color:#123c69;">
                    Loan Amount
                </h2>
                <h3 style="color:#1f2937;">
                    ₹{loan_amount:,.2f}
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 14px;
                padding: 20px;
                border: 1px solid #d8e2ef;
                margin-bottom: 15px;
            ">
                <h2 style="color:#123c69;">
                    Loan Term
                </h2>
                <h3 style="color:#1f2937;">
                    {loan_term} Months
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 14px;
                padding: 20px;
                border: 1px solid #d8e2ef;
                margin-bottom: 15px;
            ">
                <h2 style="color:#123c69;">
                    Annual Interest Rate
                </h2>
                <h3 style="color:#1f2937;">
                    {annual_interest_rate:.1f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 14px;
                padding: 20px;
                border: 1px solid #d8e2ef;
                margin-bottom: 15px;
            ">
                <h2 style="color:#123c69;">
                    Total Amount Payable
                </h2>
                <h3 style="color:#1f2937;">
                    ₹{total_amount_payable:,.2f}
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 14px;
                padding: 20px;
                border: 1px solid #d8e2ef;
                margin-bottom: 15px;
            ">
                <h2 style="color:#123c69;">
                    Total Interest
                </h2>
                <h3 style="color:#1f2937;">
                    ₹{total_interest:,.2f}
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            f"You would pay approximately "
            f"₹{monthly_emi:,.2f} every month for "
            f"{loan_term} months."
        )


        # -------------------------------------------------
        # APPLICATION SUMMARY
        # -------------------------------------------------

        st.markdown(
            '<div class="section-heading">📋 Application Summary</div>',
            unsafe_allow_html=True
        )

        summary_data = pd.DataFrame(
            {
                "Field": [
                    "Age",
                    "Monthly Income",
                    "Loan Amount",
                    "Credit Score",
                    "Employment Status",
                    "Existing Loans",
                    "Loan Term",
                    "Property Area"
                ],
                "Value": [
                    f"{age} years",
                    f"₹{monthly_income:,.2f}",
                    f"₹{loan_amount:,.2f}",
                    credit_score,
                    employment_status,
                    existing_loans,
                    f"{loan_term} months",
                    property_area
                ]
            }
        )

        st.table(summary_data)

    except Exception as error:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(error)


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

if df is not None:

    st.markdown(
        '<div class="section-heading">📈 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    total_records = len(df)

    if "LoanApproved" in df.columns:

        approved_records = int(
            df["LoanApproved"].sum()
        )

        rejected_records = total_records - approved_records

        approval_rate = (
            approved_records / total_records * 100
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Applications",
                total_records
            )

        with col2:
            st.metric(
                "Approved",
                approved_records
            )

        with col3:
            st.metric(
                "Rejected",
                rejected_records
            )

        with col4:
            st.metric(
                "Approval Rate",
                f"{approval_rate:.2f}%"
            )

        st.subheader("Loan Approval Distribution")

        distribution_data = pd.DataFrame(
            {
                "Status": [
                    "Approved",
                    "Rejected"
                ],
                "Count": [
                    approved_records,
                    rejected_records
                ]
            }
        )

        st.bar_chart(
            distribution_data.set_index("Status")
        )

    with st.expander("View Sample Dataset"):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🏦 Loan Approval Prediction System |
        Developed using Python, Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)