# 📊 Income Prediction using Machine Learning (1994 U.S. Census Data)

This project models whether an individual earns more than $50,000 per year based on demographic features collected in the 1994 U.S. Census. The task is a classic binary classification problem, relevant to domains such as targeted outreach in non-profit fundraising.

## 🎯 Objective

Construct and optimize a machine learning pipeline that:
- Predicts income class (`<=50K` or `>50K`)
- Evaluates multiple models using cross-validation
- Selects the best-performing model
- Saves the model for future deployment or inference

---

## 📁 Project Structure
finding-donors-aws-sagemaker/
│
├── census.csv # Raw input data
├── train_model.py # Main training script
├── findingDonors.joblib # Saved model (after training)
├── README.md # Project documentation
└── requirements.txt # Python dependencies (optional)


---

## 🧪 Dataset

- **Source**: [UCI Machine Learning Repository - Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
- **Target Variable**: `income` — whether a person earns `>50K` or `<=50K`
- **Features**: age, education level, hours worked, occupation, capital-gain/loss, etc.

---

## 🛠️ Features & Methods

- **Preprocessing**:
  - Log-transform on skewed features (`capital-gain`, `capital-loss`)
  - Feature scaling using `MinMaxScaler`
  - One-hot encoding for categorical variables
  - Built with `Pipeline` + `ColumnTransformer`

- **Models Compared**:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier

- **Evaluation**:
  - 5-fold Cross-Validation
  - Accuracy Score
  - F-beta Score (β=0.5)

- **Model Selection**:
  - Hyperparameter tuning with `GridSearchCV` (Random Forest)
  - Best model saved with `joblib`

---

## 🚀 How to Run

1. **Clone the repository**:

   git clone https://github.com/NorhanShaker1990/finding-donors-aws-sagemaker.git
   cd finding-donors-aws-sagemaker
   
2. **Run Training Script**:
	python train_model.py
	
3. **Output**:

	Trained model: findingDonors.joblib

	Console outputs with evaluation metrics	
	
## Sample Result
-XGBoost with Pipeline - Cross-validation accuracy: 0.8652
-Final Accuracy on Test Set: 0.8731
-F-beta Score (β=0.5): 0.7824	

	

