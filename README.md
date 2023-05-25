# Churn Prediction Analysis

This repository contains code for churn prediction analysis, which involves predicting customer churn using a classification model. The code performs data loading, preprocessing, feature engineering, and model training.

## Dataset

The analysis is performed on the "Churn_Modelling" dataset, which is stored in the "archive" directory. The dataset contains customer information such as credit score, tenure, balance, and demographics. The target variable is the "Exited" column, indicating whether a customer has churned (1) or not (0).

## Code Overview

The code is structured as follows:

1. Data Loading: The dataset is loaded from a CSV file using the pandas library.
2. Data Preprocessing: Unnecessary columns are dropped, and numerical columns are scaled using MinMaxScaler.
3. Feature Engineering: Additional categorical columns are created based on age conditions, and certain columns are converted to numerical values.
4. Model Training: The dataset is split into training and testing sets, and a classification model is trained using the TensorFlow and Keras libraries.
5. Model Evaluation: The trained model is evaluated using various metrics, such as accuracy, precision, recall, and F1-score.
6. Visualization: Confusion matrix and histograms are plotted to visualize the model's performance and analyze data distributions.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- TensorFlow
- Keras
- seaborn
- matplotlib

## Usage

1. Install the required dependencies using pip or conda: 
pip install pandas scikit-learn tensorflow keras seaborn matplotlib

kotlin
Copy code

2. Clone this repository:
git clone https://github.com/Firangi0799/churn-prediction-analysis.git

css
Copy code

3. Navigate to the project directory:
cd churn-prediction-analysis

css
Copy code

4. Run the code:
python churn_prediction.py

python
Copy code

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or enhancements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
