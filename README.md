# Telco Customer Churn — Supervised Machine Learning Project

This project presents an end-to-end supervised machine learning analysis of the **Telco Customer Churn dataset**, completed as a capstone project for my data science bootcamp.

The objective was to predict customer churn using historical account and service data, while demonstrating a **careful analytical workflow**, thoughtful feature engineering, and clear evaluation of multiple machine learning models.

Rather than treating this as a purely technical exercise, I approached the project with an emphasis on **documentation, justification of decisions, and interpretability**, reflecting how real-world machine learning work is carried out in practice. This was my first project of the bootcamp where I felt like I was operating as a data scientist.

![A pine forest in snow. The image is shown in tones of dark blue.](mono-pine-forest.png)

### What's in this repository
- **Jupyter Notebook:** `telco-customer-churn.ipynb`
- **Images:** visuals and reviewer feedback
- **Dataset:** `Telco-Customer-Churn.csv`
- **Requirements:** Python dependencies (`requirements.txt`)

---

### Project Context

Customer churn occurs when subscribers discontinue their service. For telecommunications providers, understanding which customers are likely to churn — and why — is critical for designing effective retention strategies.

This project involves:

- In-depth exploratory data analysis  
- Feature engineering and dimensional reduction  
- Handling multicollinearity and class imbalance  
- Training and evaluating supervised learning models  
- Interpreting model performance using appropriate metrics  

Two classification models were developed and compared:

- **Logistic Regression**  
- **Random Forest Classifier**

---

### Approach Overview

The analysis follows a structured machine learning workflow:

1. **Exploratory data analysis and feature inspection**
2. **Feature engineering and dimensional simplification**
3. **Correlation and multicollinearity assessment**
4. **Data scaling and train–test splitting**
5. **Model training**
6. **Model evaluation using accuracy, confusion matrices, precision, recall, and OOB error**
7. **Model comparison and selection**

Throughout the project, decisions were explicitly documented, including where features were retained for interpretive value rather than predictive strength.

---

### Feature Engineering and Data Preparation

The dataset contains a large number of categorical variables describing customer services. Rather than applying one-hot encoding indiscriminately, I evaluated the **level of granularity required** for effective prediction and interpretation.

Key feature engineering decisions included:

- Converting `SeniorCitizen` into a categorical feature for consistency
- Grouping service-level variables into higher-level indicators
- Creating simplified service groupings to reduce dimensionality
- Retaining features that provided behavioural insight even if predictive strength was limited

This approach reduced unnecessary complexity while preserving meaningful structure in the data.

---

### Exploratory Analysis and Insights

Exploratory visualisations were used to understand both customer behaviour and relationships between variables:

- **Correlation heatmaps** identified strong dependencies, particularly between tenure and total charges
- **Histograms** highlighted tenure distributions and potential churn risk windows
- **Scatter plots** confirmed expected relationships between charges
- **Box plots** revealed that churn occurs predominantly within the first 30 months of tenure

These findings supported both feature selection decisions and later model interpretation.

Variance Inflation Factor (VIF) analysis was also used to assess multicollinearity prior to modelling.

---

### Handling Class Imbalance

The target variable (`Churn`) was imbalanced.

To address this without introducing unnecessary variance or overfitting:

- Undersampling was avoided due to reduced training data
- Oversampling was avoided due to overfitting risk
- **Class weighting (`class_weight='balanced'`)** was applied instead

This ensured the models accounted for minority churn cases while preserving the full dataset.

---

## Model Development

### Logistic Regression

A logistic regression classifier was trained as a baseline model.

- Features were scaled using **min–max scaling**
- Scaling was fit on training data only to avoid leakage
- Accuracy achieved: **~0.74**

This model provided interpretability and a useful benchmark for comparison.

---

### Random Forest Classifier

A random forest model was then developed to capture non-linear relationships inherent in customer behaviour data.

Key characteristics:

- 2000 trees
- Bootstrapping enabled
- Out-of-bag (OOB) error estimation activated
- Feature importance analysis performed

Low-importance features were identified and removed, improving efficiency without degrading performance.

Final performance:

- Accuracy: **~0.77**
- OOB error closely matched validation error, indicating good generalisation

---

### Model Evaluation and Comparison

Both models were evaluated using:

- Confusion matrices
- Precision and recall
- Weighted averages for imbalanced data
- F1-scores
- OOB error (random forest)

While both models performed reasonably well, the **random forest model consistently achieved stronger recall**, which is particularly important in churn prediction.

In this context:

- **False positives** (predicting retention when churn occurs) are costly
- Recall is therefore a more appropriate decision metric than accuracy alone

Based on this evaluation, the random forest model was identified as the more suitable choice.

---

### Skills Demonstrated

- Exploratory data analysis  
- Feature engineering and dimensional reduction  
- Multicollinearity assessment (correlation and VIF)  
- Handling imbalanced datasets  
- Logistic regression classification  
- Random forest modelling and ensemble learning  
- Model evaluation using precision, recall, confusion matrices, and OOB error  
- Clear documentation and analytical reasoning  

---

### Requirements
Install the required Python dependencies with `requirements.txt`
