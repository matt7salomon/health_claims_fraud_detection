# Health Insurance Fraud Detection

This notebook contains a machine learning pipeline designed to detect potential fraud in health insurance claims data. The dataset used includes features that describe health conditions, treatments, and payments, as well as whether or not each provider is flagged as potentially fraudulent. The goal is to build a classification model to predict which providers are likely involved in fraudulent activities.

## Dataset Overview

The dataset contains information on healthcare providers, patients, treatments, chronic conditions, and billing details. The data is split into features and labels, where the features describe the attributes related to insurance claims and the labels indicate whether fraud is suspected.

### Key Features:

- **Chronic Conditions**: Multiple columns that indicate whether a patient has certain chronic conditions such as heart disease, stroke, arthritis, etc.
- **Annual Reimbursement and Deductible Amounts**: Financial data including the annual amount reimbursed and the deductible amount for both inpatient and outpatient claims.
- **Claim Dates**: The date when the claim was filed and processed.
- **Provider Information**: The provider ID and whether the provider has been flagged as potentially fraudulent.

### Target:

- **PotentialFraud**: A binary classification (`Yes`/`No`) indicating whether the provider is suspected of fraudulent activities.

---

## Methods and Functions

### 1. **Data Preprocessing**:
   - The notebook begins with loading raw CSV data and cleaning it to remove missing values, normalize formats, and preprocess text columns. It uses libraries such as `pandas` to handle the data cleaning process.

### 2. **Feature Engineering**:
   - **Chronic Conditions Features**: These are categorical values that describe a patient's chronic condition history. The categorical data is encoded to make it suitable for machine learning models.
   - **Financial Features**: Financial columns related to reimbursement and deductibles are processed as numeric values to be used directly in the model.

### 3. **Fraud Detection Model**:
   The main task of the notebook is to predict whether a healthcare provider is likely to be involved in fraud.

   #### Key Functions:

   - **generic_clf**: 
     - A helper function to build and train any classifier provided to it. The function trains the classifier on the training set, makes predictions on both the training and test sets, and calculates the error rate (misclassification rate).
     - Parameters:
       - `X_train`, `Y_train`: Training data features and labels.
       - `X_test`, `Y_test`: Test data features and labels.
       - `clf`: The classifier to be trained.
     - Returns: Training and testing error rates.

   - **catboost_clf**:
     - This function builds a CatBoost classifier to perform classification tasks. It is used to train the model using gradient boosting on decision trees, which is robust for structured data.
     - The function trains the CatBoost model and predicts whether a provider is fraudulent.
     - Parameters:
       - `M`: The number of boosting iterations (control model complexity and accuracy).
     - Returns: The error rate on both the training and test sets.

### 4. **Visualization**:
   - **plot_error_rate**: This function is used to visualize the error rates of the model as the number of iterations increases. It provides a graphical representation of how the model improves over time, showing both training and testing error rates across iterations.

---

## Fraud Detection Workflow

1. **Data Loading**:
   The dataset is loaded into the notebook using `pandas`. The training and testing sets are prepared for model evaluation, with features extracted from chronic conditions, financial data, and provider information.

2. **Model Training**:
   - Initially, a simple Decision Tree classifier is trained using the `generic_clf` function to establish a baseline performance.
   - Then, a more advanced **CatBoost** classifier is used to detect potential fraud. CatBoost is a gradient boosting algorithm that works well on categorical data and structured datasets.

3. **Evaluation**:
   The model is evaluated based on error rates (misclassification rates). The error rate is calculated for both the training and test sets, which provides insight into how well the model is performing and whether it is overfitting or underfitting.

4. **Visualization**:
   The results of the model are visualized using the `plot_error_rate` function, which plots how the error rate decreases with each iteration of the model's training process.

---

## Fraud Detection Approach

The detection of fraud in this dataset is approached as a binary classification problem. The pipeline applies a CatBoost classifier that learns from the features of insurance claims and classifies whether a provider is likely to be involved in fraud based on patterns in the data.

**Steps:**
- Preprocess the dataset to prepare features for training.
- Train a baseline Decision Tree model.
- Train and fine-tune a CatBoost model.
- Evaluate and visualize the performance of the model over several iterations.
- The final output is a classification that predicts whether each provider is involved in fraud (`Yes`/`No`).

---

## Conclusion

This notebook builds a comprehensive fraud detection pipeline that preprocesses data, trains models, and evaluates their performance. The use of the CatBoost algorithm ensures that the model can effectively handle structured data with categorical features, and the error rate analysis provides insight into how well the model generalizes to unseen data.

