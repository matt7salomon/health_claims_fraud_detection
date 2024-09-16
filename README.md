# Health Insurance Fraud Detection

This project aims to detect potential fraudulent activities in healthcare insurance claims by analyzing various aspects of patient information, medical claims, and healthcare provider behavior. The dataset comprises multiple CSV files containing information about inpatient, outpatient, and beneficiary claims. The goal is to build a machine learning model to classify healthcare providers as potentially fraudulent or not.

## Dataset Overview

The dataset consists of several files containing claims data:
- **Inpatient Claims**: Information on inpatient services and associated claims.
- **Outpatient Claims**: Information on outpatient services and associated claims.
- **Beneficiary Data**: Details about the beneficiaries, including demographic information and chronic conditions.
- **Provider Fraud Labels**: Data indicating whether a provider is flagged as potentially fraudulent.

### Key Features:
- **Claims Data**: Includes claims for inpatient and outpatient services, such as `ClmDiagnosisCode`, `ClmProcedureCode`, and amounts like `InscClaimAmtReimbursed`, `DeductibleAmtPaid`, etc.
- **Beneficiary Data**: Contains information about patients, such as age, chronic conditions, and whether the patient is deceased.
- **Fraud Labels**: The target variable (`PotentialFraud`) is binary and indicates whether a provider is involved in fraudulent activities.

---

## Methods and Functions

### 1. **Data Loading and Merging**
- The dataset is downloaded from Kaggle and consists of multiple CSV files that are read into Pandas DataFrames.
- The inpatient, outpatient, and beneficiary data are merged into a single dataset using provider IDs and patient information (`BeneID`).
  
  **Key Merging Operations:**
  - Inpatient and outpatient claims are merged on shared columns such as `ClaimID` and `Provider`.
  - Beneficiary data is merged with claims data to incorporate patient details such as age, gender, and chronic conditions.

### 2. **Feature Engineering**
   The following new features are created to improve the model's ability to detect fraud:
   
   - **Binary Encoding**: Chronic conditions are transformed into binary format (0 or 1), where 1 indicates the presence of a condition.
   - **Top Diagnosis and Procedure Codes**: The most frequent diagnosis and procedure codes are encoded into binary columns. These include common diagnosis codes like `4019` (hypertension) and procedure codes like `4019.0`.
   - **Derived Features**:
     - `Num_admit_days`: Number of days between admission and discharge for inpatient services.
     - `total_num_diag` and `total_num_proce`: Total number of diagnoses and procedures recorded for each patient.
     - **Physician Information**: Number of unique physicians attending a patient (`N_unique_Physicians`) and types of physicians (`N_Types_Physicians`).
     - **Monetary Features**: Aggregate reimbursement amounts (`total_InscClaimAmtReimbursed`) and mean reimbursement per patient (`mean_InscClaimAmtReimbursed`).

### 3. **Data Preprocessing**
   - **Normalization**: Monetary features such as `InscClaimAmtReimbursed`, `DeductibleAmtPaid`, and others are normalized to bring them into a consistent scale using the `Normalizer` class from `sklearn.preprocessing`.
   - **Handling Missing Data**: Columns with missing values are handled, such as filling NaN values in diagnosis and procedure codes with 0.
   - **Datetime Operations**: `DOB` (Date of Birth) and `ClaimStartDt` (Claim Start Date) are processed to calculate patient age at the time of the claim.

### 4. **Model Training and Evaluation**
   - **Train-Test Split**: The preprocessed dataset is split into training and test sets using the `train_test_split` method from `sklearn.model_selection`.
   - **Model Selection**: A `LightGBMClassifier` (from the LightGBM library) is trained on the training set. LightGBM is chosen for its speed and performance on tabular data.
   
   **Model Evaluation**:
   - **Confusion Matrix**: The confusion matrix is used to evaluate the performance of the model by comparing the true and predicted labels.
   - **F1 Score**: The F1 score is calculated to assess the balance between precision and recall.
   - **AUC (Area Under the Curve)**: The model's AUC score is computed to evaluate the classifier's performance in distinguishing between fraud and non-fraud cases.

### 5. **Functions in Detail**

- **`encoded_cat`**: Encodes the top 5 most frequent diagnosis and procedure codes into binary columns, creating useful features for the model to detect patterns associated with fraudulent claims.
  
- **`N_unique_values`**: Returns the number of unique values in a specific row, used to calculate the number of unique physicians attending a patient.

- **`num_col_normalizer`**: Normalizes a given column in the dataset, helping to bring numerical features onto a comparable scale.

- **`Predict_Fraud_providers`**: The core function that merges the raw datasets, processes and transforms the data, and prepares it for modeling. This function also adds new features that are critical for detecting fraudulent activity.

- **`get_error_rate`**: Computes the misclassification rate by comparing predictions to true labels.

- **`get_confusion_matrix`**: Generates a confusion matrix and visualizes it using a heatmap to assess the model's prediction accuracy.

---

## Fraud Detection Workflow

1. **Data Loading and Merging**: Raw inpatient, outpatient, and beneficiary data is merged into a single dataset for analysis.
2. **Feature Engineering**: Several new features are derived, including binary encodings of chronic conditions, the number of unique physicians, the number of diagnosis and procedure codes, and monetary features like total reimbursement.
3. **Data Preprocessing**: The dataset is cleaned and normalized. Missing values are handled appropriately, and age, admission days, and other numerical features are calculated.
4. **Model Training**: A `LightGBMClassifier` is trained to predict whether a healthcare provider is involved in fraud.
5. **Model Evaluation**: The model's performance is evaluated using a confusion matrix, F1 score, and AUC score to measure its ability to correctly identify fraudulent providers.

---

## Conclusion

This notebook builds a comprehensive fraud detection pipeline that integrates multiple data sources, performs feature engineering, and trains a machine learning model to predict healthcare provider fraud. By using key features such as diagnosis codes, procedure codes, reimbursement amounts, and physician data, the model is able to make accurate predictions about fraudulent activities.
