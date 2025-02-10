# Churn Prediction Project

## Project Overview
This project focuses on building a machine learning model to predict customer churn for a business. Churn prediction is a crucial aspect of customer relationship management, helping companies retain customers by identifying those likely to leave.

### Objective
The primary objective of this project is to develop a predictive model using customer data to accurately determine whether a customer will churn or remain with the company.

## Dataset Description
The dataset used for this project contains the following features:

| Feature Name        | Description |
|---------------------|-------------|
| **RowNumber**       | Row index in the dataset for reference |
| **CustomerId**      | Unique identifier for the customer |
| **Surname**         | Customer's surname |
| **CreditScore**     | Customer's credit score |
| **Geography**       | Customer's country (Geographical location) |
| **Gender**          | Customer's gender |
| **Age**             | Customer's age |
| **Tenure**          | Number of years the customer has been with the company |
| **Balance**         | Customer's bank account balance |
| **NumOfProducts**   | Number of products the customer uses |
| **HasCrCard**       | Whether the customer has a credit card (1 for Yes, 0 for No) |
| **IsActiveMember**  | Whether the customer is an active member (1 for Yes, 0 for No) |
| **EstimatedSalary** | Estimated annual salary of the customer |
| **Exited**          | Target feature: 1 if the customer has churned, 0 otherwise |

## Project Structure
The project contains the following key files and directories:

- **data/**: Contains the dataset.
- **model/**: Stores the trained model file `model.pkl`.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and model development.
- **scripts/**: Python scripts for data preprocessing, model training, and evaluation.
- **README.md**: This documentation file.

## Model Training and Evaluation
The workflow of the project includes:

1. **Data Preprocessing:**
   - Handle missing values and outliers.
   - Encode categorical features.
   - Normalize numerical features where necessary.

2. **Model Development:**
   - Multiple algorithms were tested, such as Logistic Regression, Decision Trees, and Gradient Boosting.
   - The best-performing model was selected and saved using the `pickle` library.

3. **Model Evaluation:**
   - Metrics such as accuracy, precision, recall, and F1-score were used to assess the model's performance.


## Model Loading and Usage
The trained model is saved as `model.pkl`. Below is an example code to load and use the model for predictions:

```python
import pickle

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Example usage: Predicting churn
sample_data = [[600, 'France', 'Female', 40, 3, 60000, 2, 1, 0, 50000]]
result = model.predict(sample_data)
print("Churn Prediction:", result)
```

## Conclusion
This project demonstrates an end-to-end pipeline for building a churn prediction system. The model can help businesses proactively engage with customers who are likely to churn, improving retention rates and overall customer satisfaction.

