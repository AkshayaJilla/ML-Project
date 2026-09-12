#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


np.random.seed(42)
n=2000
applicant_id = np.arange(100001, 100001 + n)
print(applicant_id[:10])
gender = np.random.choice(
    ["Male", "Female"],
    size=n
)
print(gender[:10])
age = np.random.randint(
    21,
    61,
    size=n
)
print(age[:10])
employment_status = np.random.choice(
    ["Employed", "Self-Employed", "Unemployed"],
    size=n,
    p=[0.60, 0.30, 0.10]
)
print(employment_status[:10])


# In[3]:


income_base = (
    18000
    + (age - 21) * 1100
    + np.where(employment_status == "Self-Employed", 9000, 0)
    + np.where(employment_status == "Unemployed", -9000, 0)
)

monthly_income = (
    income_base + np.random.normal(0, 12000, n)
)

monthly_income = np.maximum(
    7000,
    monthly_income
).round(0)

print(monthly_income[:10])


# In[4]:


credit_score = (
    520
    + (monthly_income - 15000) / 180
    + np.where(
        employment_status == "Employed",
        25,
        0
    )
    + np.where(
        employment_status == "Self-Employed",
        10,
        0
    )
    + np.random.normal(0, 55, n)
)

credit_score = np.clip(
    credit_score,
    300,
    850
).round(0)

print(credit_score[:10])


# In[5]:


existing_loans = np.random.poisson(
    1.2,
    n
)

existing_loans = np.clip(
    existing_loans,
    0,
    6
)

print(existing_loans[:10])
loan_amount = (
    monthly_income *
    np.random.uniform(2.0, 7.0, n)
    + np.random.normal(0, 25000, n)
)

loan_amount = np.maximum(
    20000,
    loan_amount
).round(0)

print(loan_amount[:10])
loan_term = np.random.choice(
    [12, 24, 36, 48, 60],
    size=n,
    p=[0.10, 0.20, 0.35, 0.20, 0.15]
)

print(loan_term[:10])


# In[6]:


property_area = np.random.choice(
    ["Urban", "Semi-Urban", "Rural"],
    size=n,
    p=[0.45, 0.35, 0.20]
)
print(property_area[:10])


# In[8]:


income_score = np.clip(
    (monthly_income - 15000) / 60000,
    -1,
    1
)

credit_score_norm = (
    credit_score - 600
) / 100

loan_burden = (
    loan_amount /
    np.maximum(monthly_income, 1)
)

loan_burden_score = np.clip(
    (4.5 - loan_burden) / 3.0,
    -1.5,
    1.5
)
employment_score = np.select(
    [
        employment_status == "Employed",
        employment_status == "Self-Employed",
        employment_status == "Unemployed"
    ],
    [
        0.55,
        0.25,
        -1.25
    ],
    default=0
)
existing_loan_score = np.clip(
    (3 - existing_loans) / 3,
    -1,
    1
)

print(existing_loan_score[:10])


# In[9]:


logit = (
    -0.25
    + 1.25 * income_score
    + 1.55 * credit_score_norm
    + 0.95 * loan_burden_score
    + employment_score
    + existing_loan_score
    + np.random.normal(0, 0.55, n)
)
approval_probability = (
    1 / (1 + np.exp(-logit))
)
loan_approved = np.random.binomial(
    1,
    approval_probability
)


# In[10]:


df = pd.DataFrame({
    "ApplicantID": applicant_id,
    "Gender": gender,
    "Age": age,
    "MonthlyIncome": monthly_income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "EmploymentStatus": employment_status,
    "ExistingLoans": existing_loans,
    "LoanTerm": loan_term,
    "PropertyArea": property_area,
    "LoanApproved": loan_approved
})
df.head()


# In[15]:


# ==========================================
# STEP 8: DATA CLEANING AND VALIDATION
# ==========================================

# 1. Check dataset information
print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

# 2. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 3. Handle missing numerical values
numeric_columns = [
    "Age",
    "MonthlyIncome",
    "LoanAmount",
    "CreditScore",
    "ExistingLoans",
    "LoanTerm"
]

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# 4. Handle missing categorical values
categorical_columns = [
    "Gender",
    "EmploymentStatus",
    "PropertyArea"
]

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )

# 5. Remove duplicate rows
df = df.drop_duplicates()

# 6. Validate age
df.loc[
    (df["Age"] < 18) | (df["Age"] > 100),
    "Age"
] = np.nan

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

# 7. Validate credit score
df.loc[
    (df["CreditScore"] < 300) |
    (df["CreditScore"] > 850),
    "CreditScore"
] = np.nan

df["CreditScore"] = df["CreditScore"].fillna(
    df["CreditScore"].median()
)

# 8. Validate income
df.loc[
    df["MonthlyIncome"] < 0,
    "MonthlyIncome"
] = np.nan

df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
    df["MonthlyIncome"].median()
)

# 9. Validate loan amount
df.loc[
    df["LoanAmount"] < 0,
    "LoanAmount"
] = np.nan

df["LoanAmount"] = df["LoanAmount"].fillna(
    df["LoanAmount"].median()
)

# 10. Final checks
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print(
    "\nDuplicate rows after cleaning:",
    df.duplicated().sum()
)

print("\nFinal dataset shape:")
print(df.shape)

# 11. Save cleaned dataset
df.to_csv(
    "loan_applications.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# In[12]:


print(df["LoanApproved"].value_counts())


# In[13]:


df.info()


# In[14]:


df.isnull().sum()


# In[19]:


df.head()


# In[21]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
df.describe()


# In[23]:


approval_counts = df["LoanApproved"].value_counts()
print(approval_counts)
approval_percentage = (
    df["LoanApproved"]
    .value_counts(normalize=True)
    * 100
)
print(approval_percentage)
income_analysis = df.groupby(
    "LoanApproved"
)["MonthlyIncome"].mean()

print(income_analysis)
print(
    "Average income of rejected applicants:",
    df[df["LoanApproved"] == 0]["MonthlyIncome"].mean()
)

print(
    "Average income of approved applicants:",
    df[df["LoanApproved"] == 1]["MonthlyIncome"].mean()
)


# In[26]:


credit_analysis = df.groupby(
    "LoanApproved"
)["CreditScore"].mean()

print(credit_analysis)
employment_analysis = df.groupby(
    "EmploymentStatus"
)["LoanApproved"].mean()

print(employment_analysis)
property_analysis = (
    df.groupby("PropertyArea")["LoanApproved"]
    .mean()
    * 100
)
print(property_analysis)
loan_analysis = df.groupby(
    "ExistingLoans"
)["LoanApproved"].mean() * 100

print(loan_analysis)


# In[25]:


employment_approval_rate = (
    df.groupby("EmploymentStatus")["LoanApproved"]
    .mean()
    * 100
)

print(employment_approval_rate)


# In[29]:


numeric_columns = [
    "Age",
    "MonthlyIncome",
    "LoanAmount",
    "CreditScore",
    "ExistingLoans",
    "LoanTerm",
    "LoanApproved"
]
correlation_matrix = df[
    numeric_columns
].corr()

print(correlation_matrix)


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="EmploymentStatus",
    hue="LoanApproved"
)

plt.title("Loan Approvals by Employment Status")
plt.xlabel("Employment Status")
plt.ylabel("Number of Applicants")

plt.show()


# In[31]:


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# In[32]:


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="MonthlyIncome",
    y="LoanAmount",
    hue="LoanApproved"
)

plt.title(
    "Monthly Income vs Loan Amount"
)

plt.xlabel("Monthly Income")
plt.ylabel("Loan Amount")

plt.show()


# In[33]:


approval_counts = df[
    "LoanApproved"
].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    approval_counts,
    labels=["Approved", "Rejected"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Loan Approval Distribution")

plt.show()


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
import numpy as np

df = pd.read_csv("loan_applications.csv")

print(df.head())
print(df.shape)


# In[37]:


X = df.drop(
    columns=["ApplicantID", "LoanApproved"]
)

y = df["LoanApproved"]

print("X shape:", X.shape)
print("y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
numeric_features = [
    "Age",
    "MonthlyIncome",
    "LoanAmount",
    "CreditScore",
    "ExistingLoans",
    "LoanTerm"
]

print(numeric_features)
categorical_features = [
    "Gender",
    "EmploymentStatus",
    "PropertyArea"
]

print(categorical_features)


# In[38]:


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
classifier = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])
model.fit(X_train, y_train)

print("Model training completed successfully!")
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:20])


# In[39]:


comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(20))


# In[40]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[42]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[45]:


def predict_loan():
    print("===== Loan Approval Prediction =====")

    try:
        age = int(input("Enter Age: "))
        income = float(input("Enter Monthly Income: "))
        loan_amount = float(input("Enter Loan Amount: "))
        credit_score = int(input("Enter Credit Score: "))

        employment = input(
            "Enter Employment Status (Employed/Self-Employed/Unemployed): "
        )

        existing_loans = int(input("Enter Number of Existing Loans: "))
        loan_term = int(input("Enter Loan Term (12/24/36/48/60): "))

        property_area = input(
            "Enter Property Area (Urban/Semi-Urban/Rural): "
        )

        # Validate values
        if not 21 <= age <= 60:
            raise ValueError("Age must be between 21 and 60.")

        if income <= 0:
            raise ValueError("Monthly income must be greater than 0.")

        if loan_amount <= 0:
            raise ValueError("Loan amount must be greater than 0.")

        if not 300 <= credit_score <= 850:
            raise ValueError("Credit score must be between 300 and 850.")

        if employment not in ["Employed", "Self-Employed", "Unemployed"]:
            raise ValueError("Invalid employment status.")

        if existing_loans < 0:
            raise ValueError("Existing loans cannot be negative.")

        if loan_term not in [12, 24, 36, 48, 60]:
            raise ValueError("Loan term must be 12, 24, 36, 48 or 60.")

        if property_area not in ["Urban", "Semi-Urban", "Rural"]:
            raise ValueError("Invalid property area.")

        # Create input data
        new_applicant = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [income],
            "LoanAmount": [loan_amount],
            "CreditScore": [credit_score],
            "EmploymentStatus": [employment],
            "ExistingLoans": [existing_loans],
            "LoanTerm": [loan_term],
            "PropertyArea": [property_area],
            "Gender": ["Male"]
        })

        # Make prediction
        prediction = model.predict(new_applicant)[0]

        print("\n===== Prediction Result =====")

        if prediction == 1:
            print("Loan Status: APPROVED")
        else:
            print("Loan Status: REJECTED")

    except ValueError as e:
        print("Invalid input:", e)
predict_loan()


# In[47]:


import joblib
joblib.dump(model, "loan_approval_model.pkl")

print("Model saved successfully!")
joblib.dump(model, "loan_approval_model.pkl")
loaded_model = joblib.load("loan_approval_model.pkl")
print("Model loaded successfully!")


# In[48]:


test_prediction = loaded_model.predict(X_test)

print("Prediction successful!")
print(test_prediction[:10])


# In[ ]:




