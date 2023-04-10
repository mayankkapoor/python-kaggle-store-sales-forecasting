import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from math import sqrt, log

def rmsle(y_true, y_pred):
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return np.sqrt(np.mean(log_diff**2))

def visualize_sales_over_time(train_df):
    plt.figure(figsize=(15, 5))
    print(train_df.columns)
    plt.plot(train_df['date'], train_df['sales'])
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales over Time")
    plt.show()

def find_best_sarimax_parameters(train_data):
    # Example SARIMAX parameters
    p = range(0, 3)
    d = range(1, 2)
    q = range(0, 3)
    P = range(0, 2)
    D = range(1, 2)
    Q = range(0, 2)
    s = 7
    # Generate all combinations of p, d, q, P, D, Q
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in itertools.product(P, D, Q)]

    # Record the AIC and corresponding parameters for each model
    aic_results = []
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train_data['sales'],
                                                order=param,
                                                seasonal_order=seasonal_param,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                aic_results.append((param, seasonal_param, results.aic))
            except:
                continue

    # Find the best model based on the lowest AIC value
    best_model = min(aic_results, key=lambda x: x[2])
    print(f"Best SARIMAX parameters: {best_model[0]}, seasonal parameters: {best_model[1]}, AIC: {best_model[2]}")
    return best_model[0], best_model[1]

def train_models(train_data):
    # Train SARIMAX model using the best parameters
    model_sarimax = SARIMAX(train_data['sales'], order=best_order, seasonal_order=best_seasonal_order)
    results_sarimax = model_sarimax.fit()

    # Train the Exponential Smoothing model
    model_exp_smoothing = ExponentialSmoothing(train_data['sales'], seasonal_periods=m, trend='add', seasonal='add')
    results_exp_smoothing = model_exp_smoothing.fit()

    # Train the Prophet model
    train_data_prophet = train_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model_prophet.fit(train_data_prophet)

    return results_sarimax, results_exp_smoothing, model_prophet

def evaluate_models(validation_data, results_sarimax, results_exp_smoothing, model_prophet):
    validation_data['forecast_sarimax'] = results_sarimax.predict(start='2016-11-25', end='2017-08-15')
    validation_data['forecast_exp_smoothing'] = results_exp_smoothing.predict(start='2016-11-25', end='2017-08-15')

    # Evaluate the Prophet model
    validation_data_prophet = validation_data[['Date']].rename(columns={'Date': 'ds'})
    forecast_prophet = model_prophet.predict(validation_data_prophet)
    validation_data['forecast_prophet'] = forecast_prophet['yhat'].values

    # Calculate RMSLE for each model
    rmsle_sarimax = rmsle(validation_data['Sales'], validation_data['forecast_sarimax'])
    rmsle_exp_smoothing = rmsle(validation_data['Sales'], validation_data['forecast_exp_smoothing'])
    rmsle_prophet = rmsle(validation_data['Sales'], validation_data['forecast_prophet'])

    return rmsle_sarimax, rmsle_exp_smoothing, rmsle_prophet

def save_best_model(best_model, results_sarimax, results_exp_smoothing, model_prophet):
    if best_model[0] == 'SARIMAX':
        best_model_instance = results_sarimax
    elif best_model[0] == 'Exponential Smoothing':
        best_model_instance = results_exp_smoothing
    else:  # Prophet
        best_model_instance = model_prophet

    with open('best_model.pkl', 'wb') as file:
        pickle.dump(best_model_instance, file)

def generate_submission(test_df, best_model, results_sarimax, results_exp_smoothing, model_prophet):
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    if best_model[0] == 'SARIMAX':
        test_df['forecast'] = results_sarimax.predict(start='YYYY-MM-DD', end='YYYY-MM-DD')
    elif best_model[0] == 'Exponential Smoothing':
        test_df['forecast'] = results_exp_smoothing.predict(start='YYYY-MM-DD', end='YYYY-MM-DD')
    else:  # Prophet
        test_data_prophet = test_df[['Date']].rename(columns={'Date': 'ds'})
        forecast_prophet = model_prophet.predict(test_data_prophet)
        test_df['forecast'] = forecast_prophet['yhat'].values

    # Generate submission file
    submission_df = test_df[['Id', 'forecast']]
    submission_df.columns = ['Id', 'Sales']
    submission_df.to_csv("submission.csv", index=False)

def main():
    # Step 1: Data Exploration
    train_df = pd.read_csv("train.csv")
    store_df = pd.read_csv("stores.csv")

    # Analyze and visualize the data
    print(train_df.head())
    print(store_df.head())

    # Check for missing values
    print(train_df.isnull().sum())
    print(store_df.isnull().sum())

    # visualize_sales_over_time(train_df) # This isn't working

    # Step 2: Data Preprocessing
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df.sort_values(by='date', inplace=True)

    # Merge the train and store data
    merged_df = pd.merge(train_df, store_df, on='store_nbr')

    # Step 3: Feature Engineering
    merged_df['DayOfWeek'] = merged_df['date'].dt.dayofweek
    merged_df['Month'] = merged_df['date'].dt.month

    # Step 4: Model Selection

    # Step 5: Model Training
    # A good value for YYYY-MM-DD dataset split is 2016-11-25 which gives an 80-20 split
    # with the dataset ranging from January 1, 2013, to August 15, 2017
    split_date = '2016-11-25'
    train_data = merged_df.loc[merged_df['date'] < split_date]
    validation_data = merged_df.loc[merged_df['date'] >= split_date]

    # Find the best SARIMAX parameters using grid search
    best_order, best_seasonal_order = find_best_sarimax_parameters(train_data)

    # Train the models with the best SARIMAX parameters
    results_sarimax, results_exp_smoothing, model_prophet = train_models(train_data, best_order, best_seasonal_order)

    # Step 6: Model Evaluation
    rmsle_sarimax, rmsle_exp_smoothing, rmsle_prophet = evaluate_models(validation_data, results_sarimax, results_exp_smoothing, model_prophet)

    print(f'RMSLE (SARIMAX): {rmsle_sarimax}')
    print(f'RMSLE (Exponential Smoothing): {rmsle_exp_smoothing}')
    print(f'RMSLE (Prophet): {rmsle_prophet}')

    # Choose the best model based on RMSLE
    best_model = min(('SARIMAX', rmsle_sarimax), ('Exponential Smoothing', rmsle_exp_smoothing), ('Prophet', rmsle_prophet), key=lambda x: x[1])
    print(f'Best model: {best_model[0]} with RMSLE: {best_model[1]}')

    save_best_model(best_model, results_sarimax, results_exp_smoothing, model_prophet)

    # Step 8: Forecasting
    test_df = pd.read_csv("test.csv")
    generate_submission(test_df, best_model, results_sarimax, results_exp_smoothing, model_prophet)

if __name__ == "__main__":
    main()
