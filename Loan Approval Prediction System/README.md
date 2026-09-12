# 🏦 Loan Approval Prediction System

## 📌 Project Overview

The Loan Approval Prediction System is a Machine Learning project that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant and financial information.

The project demonstrates a complete Machine Learning workflow including:

- Dataset generation
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature encoding
- Machine Learning model training
- Model evaluation
- Loan approval prediction
- Interactive Streamlit web application

---

## 🎯 Objective

The main objective of this project is to build a Machine Learning system that can predict loan approval based on factors such as:

- Applicant age
- Monthly income
- Loan amount
- Credit score
- Employment status
- Existing loans
- Loan term
- Property area
- Gender

The system helps demonstrate how Machine Learning can be applied to a real-world loan approval problem.

---

## 📊 Dataset

The project uses a synthetic dataset containing **2,000 loan applications**.

### Dataset Features

| Feature | Description |
|---|---|
| ApplicantID | Unique applicant identification number |
| Gender | Gender of the applicant |
| Age | Age of the applicant |
| MonthlyIncome | Monthly income of the applicant |
| LoanAmount | Requested loan amount |
| CreditScore | Credit score of the applicant |
| EmploymentStatus | Employment type |
| ExistingLoans | Number of existing loans |
| LoanTerm | Loan duration in months |
| PropertyArea | Property location type |
| LoanApproved | Target variable (0 = Rejected, 1 = Approved) |

The dataset is stored in:

```text
loan_applications.csv