# Netflix Stock Price Prediction



## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Files Included](#files-included)
- [Project Overview](#project-overview)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [References](#references)

## Introduction
This project aims to predict the stock price of Netflix using historical data. The dataset is obtained from the Yahoo Finance API and includes information such as opening price, closing price, volume, etc. The analysis and prediction are performed using Python and various machine learning techniques, including Linear Regression, Random Forest, Gradient Boosting, and XGBoost. The project also includes exploratory data analysis (EDA) to understand the data distribution and relationships between features. The final model is selected based on performance metrics such as R2 score, rmse, and cross-validation scores. The project provides a good starting point for predicting stock prices using machine learning techniques and demonstrates the process of data collection, preprocessing, model selection, evaluation, and prediction. The code is written in a Jupyter notebook and includes detailed explanations of each step in the process. The project also includes an analysis report generated using the ProfileReport library, which provides an overview of the dataset's characteristics and distributions.

## Requirements
- Python 3.10 or higher
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, yfinance, pandas-profiling

## Files Included
- `Netflix_Stock_Price_Prediction.ipynb`: Jupyter notebook containing the code for data collection, preprocessing, EDA, model selection, evaluation, and prediction.

- `Analysis.html`: Analysis report generated using the ProfileReport library, providing an overview of the dataset's characteristics and distributions.

## Project Overview
The project is divided into the following sections:

1. **Data Collection**: The historical stock data for Netflix is obtained using the Yahoo Finance API.
    - The data includes information such as Date, Open, High, Low, Close, Dividends, Stock Splits and Volume.
    - The dataset contains 3776 rows and 8 columns, with daily stock prices from 2009 to 2024.

2. **Data Preprocessing**: 
    - The data is loaded into a DataFrame. 
    - Feature engineering is performed, including creating columns for year, month, day, and quarter end using datetime information.
    - Scaling is applied to the volume column using StandardScaler and to other features using MinMaxScaler to standardize the data.

3. **Exploratory Data Analysis (EDA)**:
    - Statistical summaries and visualizations are generated to understand the data distribution and relationships between features.

4. **Model Selection and Evaluation**:
    - Four models are chosen for prediction: Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
    - The performance of each model is evaluated using metrics such as R2 score,rmse and cross-validation scores.
    - Surprisingly, Linear Regression performs exceptionally well with an R2 score close to 1, indicating a strong correlation between predicted and actual values.
    - The Random Forest, Gradient Boosting, and XGBoost models also perform well but not as good as Linear Regression.
    - The Linear Regression model is selected as the final model for prediction.

5. **Prediction**:
    - The chosen model (Linear Regression) is used to predict the closing stock price for the next day based on the previous day's data.
    - The model is trained on 75% of the data and tested on the remaining 25%.
6. **LSTM Model (Optional)**:
    - An attempt is made to use Long Short-Term Memory (LSTM) for prediction, but the results are not satisfactory, possibly due to the limited amount of data or other factors.
    - So, the LSTM model is not selected as the final model for prediction.

## Conclusion
- The project demonstrates the process of predicting Netflix stock prices using machine learning techniques.
- The data is collected from Yahoo Finance API and preprocessed for analysis.
- Exploratory Data Analysis (EDA) is performed to understand the data distribution and relationships between features.
- Four models are evaluated for prediction: Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
- The chosen model, Linear Regression, performs exceptionally well in this scenario.
- The model is used to predict the closing stock price for the next day based on the previous day's data.
- The LSTM model is also attempted for prediction but does not perform as well as the Linear Regression model.
- The project provides a good starting point for predicting stock prices using machine learning techniques.
- Future improvements could include gathering more data, trying different feature engineering techniques, or experimenting with other advanced models.

## Acknowledgements
- The dataset used in this project is obtained from the Yahoo Finance API.
- The project is inspired by various tutorials and resources on stock price prediction using machine learning.
- The code and analysis are written by me, but I acknowledge the contributions of the open-source community and online resources.

## Contact
- Email: kirtipogra@gmail.com
- LinkedIn: [Kirti Pogra](https://www.linkedin.com/in/kirti-pogra-7b0b3b1b3/)


## References
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [Pandas Profiling Documentation](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

- [Stack Overflow](https://stackoverflow.com/)


Thank you for your interest in this project! Feel free to reach out for any questions or further details. Happy predicting!

