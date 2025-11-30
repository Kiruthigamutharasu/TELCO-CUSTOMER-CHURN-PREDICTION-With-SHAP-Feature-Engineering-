# Telco Customer Churn Prediction

### Deep Learning Model with Feature Engineering and SHAP Explainability

## Project Overview

This project builds a neural network model to predict customer churn using the Telco Customer Churn dataset. The workflow includes data preprocessing, feature engineering, ANN model training, performance evaluation, and SHAP-based model explainability.

## Dataset

The dataset used is the Telco Customer Churn dataset provided by IBM.
File required: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Methodology

### 1. Data Preprocessing

* Removed the `customerID` column
* Replaced empty values and handled missing data
* Converted `TotalCharges` to numeric
* Filled missing values with median
* Encoded categorical variables using LabelEncoder
* Standardized features with StandardScaler
* Train-test split with stratification

### 2. Feature Engineering

* `tenure_bin`: Binned tenure into four groups
* `avg_charge`: Calculated average charge as `TotalCharges / (tenure + 1)`
* Additional derived and encoded features were added to improve model performance

### 3. Model Architecture

A Sequential Neural Network built using TensorFlow/Keras:

* Dense(128, activation='relu')
* Dropout(0.3)
* Dense(64, activation='relu')
* Dropout(0.2)
* Dense(1, activation='sigmoid')

Optimizer: Adam (learning rate 0.001)
Loss: Binary Crossentropy
Metrics: Accuracy

### 4. Model Training and Evaluation

* Trained for 20 epochs
* Tracked accuracy and loss for both training and validation sets
* Final test accuracy achieved: approximately 78%

### 5. Explainability (SHAP)

SHAP was used to interpret model predictions.
Generated:

* SHAP values
* SHAP summary plot
* Feature importance insights

### 6. Model Saving

The trained model is saved as:

```
churn_model_with_features_and_shap.h5
```

## Requirements

```
tensorflow
pandas
numpy
matplotlib
scikit-learn
shap
```

## How to Run

1. Upload the dataset to the working directory
2. Run the notebook or script in Google Colab or any Python environment
3. Train the model
4. View the accuracy, loss plots, and SHAP explainability outputs
5. Use the saved model for future inference
