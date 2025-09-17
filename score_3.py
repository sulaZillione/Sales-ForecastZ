import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.stl import STLForecast
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA  # Importing the ARIMA model
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

def init():
    pass
    
def run(data):
    try:
        # Load the input data (JSON format)
        input_data = json.loads(data)

        # Convert the input data to a DataFrame
        web_service_input_data = input_data['Inputs']['web_service_input']

        Horizon = input_data['Inputs']['input1'][0]['horizon']

        Seasonality = input_data['Inputs']['input1'][0]['seasonality']

        # Convert the list of dictionaries to a DataFrame
        filtered_df1 = pd.DataFrame(web_service_input_data)
        
        filtered_df1['DateKey'] = pd.to_datetime(filtered_df1['DateKey'])

        grouping_parameter = input_data['Inputs']['input1'][0]['grouping_parameter']

        Model = input_data['Inputs']['input1'][0]['Model'] 

        parameter1 = input_data['Inputs']['input1'][0]['parameter1'] 
 

        # Create separate columns for Year, MonthNumber, QuarterNumber, and WeekNumber
        filtered_df1['Year'] = filtered_df1['DateKey'].dt.year

        filtered_df1['MonthNumber'] = filtered_df1['DateKey'].dt.month

        filtered_df1['QuarterNumber'] = filtered_df1['DateKey'].dt.quarter

        filtered_df1['WeekNumber'] = filtered_df1['DateKey'].dt.isocalendar().week

        filtered_df1['DayNumber'] = filtered_df1['DateKey'].dt.day

        # Create a new column for the selected parameter (year, quarter, month, or week)
        if grouping_parameter == 'Y':
            filtered_df1['parameter'] = filtered_df1['DateKey'].dt.to_period('Y').dt.start_time
        elif grouping_parameter == 'Q':
            filtered_df1['parameter'] = filtered_df1['DateKey'].dt.to_period('Q').dt.start_time
        elif grouping_parameter == 'M':
            filtered_df1['parameter'] = filtered_df1['DateKey'].dt.to_period('M').dt.start_time
        elif grouping_parameter == 'W':
            filtered_df1['parameter'] = filtered_df1['DateKey'].dt.to_period('W').dt.start_time
        elif grouping_parameter == 'D':
            filtered_df1['parameter'] = filtered_df1['DateKey'].dt.to_period('D').dt.start_time
        else:
            raise ValueError("Invalid grouping_parameter")

        group_cols = ['parameter']

        # Group by the specified columns and sum the 'TransactionQty' values
        summary_df = filtered_df1.groupby(group_cols)['TransactionQty'].sum().reset_index()

        # Preprocess the data (Add some scaling method like min-max scaler to minimize the outliers)
        summary_df['parameter'] = pd.to_datetime(summary_df['parameter'])

        filtered_df2 = summary_df[['parameter', 'TransactionQty']]

        filtered_df2.set_index('parameter', inplace=True)

        # Define the grouping parameter ('Y' for year, 'Q' for quarter, 'M' for month, 'W' for week)

        # Resample the data based on the grouping parameter and fill missing values
        if grouping_parameter == 'Y':
            resampled_df = filtered_df2.resample('YS').sum().fillna(0)
        elif grouping_parameter == 'Q':
            resampled_df = filtered_df2.resample('QS').sum().fillna(0)
        elif grouping_parameter == 'M':
            resampled_df = filtered_df2.resample('MS').sum().fillna(0)
        elif grouping_parameter == 'W':
            resampled_df = filtered_df2.resample('W').sum().fillna(0)
        elif grouping_parameter == 'D':
            resampled_df = filtered_df2.resample('D').sum().fillna(0)
        else:
            raise ValueError("Invalid grouping parameter. Choose 'Y', 'Q', 'M', 'W', or 'D'.")

        # Replace NaN values with 0 in the 'Quantity' column
        resampled_df['TransactionQty'] = resampled_df['TransactionQty'].fillna(0)

        # Round values in 'Quantity' column to two decimal points
        resampled_df['TransactionQty'] = resampled_df['TransactionQty'].round(2)

        # Check if we have enough data points for meaningful forecasting
        if len(resampled_df) < 2:
            return {
                "error": "Insufficient data points for forecasting. Need at least 2 data points."
            }

        # Splitting the dataset into train and test sets
        train_end = max(1, int(len(resampled_df) * 0.8))  # Ensure at least 1 point for training
        test_end = len(resampled_df)

        train_data = resampled_df.iloc[:train_end]
        test_data = resampled_df.iloc[train_end:test_end] if len(resampled_df) > train_end else resampled_df.iloc[-1:]

        # Helper function to safely calculate accuracy
        def safe_accuracy_calculation(actual, predicted):
            if len(actual) == 0 or len(predicted) == 0:
                return 0, 0
            
            mae = mean_absolute_error(actual, predicted)
            mean_actual = np.mean(actual)
            
            if mean_actual == 0 or np.isnan(mean_actual) or np.isinf(mean_actual):
                return 0, mae
            
            accuracy = max(0, 100 - abs((mae / mean_actual) * 100))
            return int(accuracy), int(mae)

        # Helper function to generate forecast dates
        def generate_forecast_dates(last_date, horizon, grouping_param):
            dates = []
            current_date = last_date
            
            for i in range(horizon):
                if grouping_param == 'Y':
                    current_date = current_date + pd.DateOffset(years=1)
                elif grouping_param == 'Q':
                    current_date = current_date + pd.DateOffset(months=3)
                elif grouping_param == 'M':
                    current_date = current_date + pd.DateOffset(months=1)
                elif grouping_param == 'W':
                    current_date = current_date + pd.DateOffset(weeks=1)
                elif grouping_param == 'D':
                    current_date = current_date + pd.DateOffset(days=1)
                
                dates.append(current_date)
            
            return dates

        if Model == "AR":
            try:
                ar_model = AutoReg(resampled_df['TransactionQty'], lags=parameter1)
                ar_results = ar_model.fit()
                
                # Generate forecast dates
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)
                
                # Forecast using AR model
                ar_forecast = ar_results.predict(start=len(resampled_df), end=len(resampled_df) + Horizon - 1, dynamic=False)
                
                # Calculate accuracy if we have test data
                if len(test_data) > 0:
                    ar_model_s = AutoReg(train_data['TransactionQty'], lags=parameter1)
                    ar_results_s = ar_model_s.fit()
                    ar_forecast_s = ar_results_s.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
                    accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], ar_forecast_s)
                else:
                    accuracy, mae = 0, 0

                best_forecast = ar_forecast
                best_model_name = "AR"

                forecast_dict = [
                    {
                        "Date": date.strftime('%Y-%m-%d'),
                        "TransactionQty": max(0, int(value)),  # Ensure non-negative values
                        "Sigma": 0
                    }
                    for date, value in zip(forecast_dates, best_forecast)
                ]

                output_dict = {
                    "Forecast": forecast_dict,
                    "BestModel": [
                        {
                            "Best Model": best_model_name,
                            "Accuracy": accuracy,
                            "MAE": mae,
                            "Parameter1": parameter1,
                            "Parameter2": 0,
                            "Parameter3": 0,
                            "Parameter4": 0,
                            "Parameter5": 0,
                            "Parameter6": 0
                        }
                    ]
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"AR Model Error: {str(e)}"}

        elif Model == "ES":
            try:
                model_s = sm.tsa.ExponentialSmoothing(resampled_df['TransactionQty'], trend=None, seasonal=None, initialization_method="estimated")
                model_results_s = model_s.fit(smoothing_level=parameter1)
                
                # Generate forecast dates
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)
                
                # Forecast using the model
                forecast_s = model_results_s.forecast(steps=Horizon)
                
                # Calculate accuracy if we have test data
                if len(test_data) > 0:
                    model = sm.tsa.ExponentialSmoothing(train_data['TransactionQty'], trend=None, seasonal=None, initialization_method="estimated")
                    model_results = model.fit(smoothing_level=parameter1)
                    forecast = model_results.forecast(steps=len(test_data))
                    accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], forecast)
                else:
                    accuracy, mae = 0, 0

                best_model_name = "ES"
                best_forecast = forecast_s

                forecast_dict = [
                    {
                        "Date": date.strftime('%Y-%m-%d'),
                        "TransactionQty": max(0, int(value)),  # Ensure non-negative values
                        "Sigma": 0
                    }
                    for date, value in zip(forecast_dates, best_forecast)
                ]
                
                output_dict = {
                    "Forecast": forecast_dict,
                    "BestModel": [
                        {
                            "Best Model": best_model_name,
                            "Accuracy": accuracy,
                            "MAE": mae,
                            "Parameter1": parameter1,
                            "Parameter2": 0,
                            "Parameter3": 0,
                            "Parameter4": 0,
                            "Parameter5": 0,
                            "Parameter6": 0
                        }
                    ]
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"ES Model Error: {str(e)}"}

        elif Model == "SARIMAX":
            try:
                # Check if we have enough data for SARIMAX
                if len(train_data) < max(Seasonality, 10):
                    return {"error": f"Insufficient data for SARIMAX model. Need at least {max(Seasonality, 10)} data points."}
                
                sarimax_model = SARIMAX(train_data['TransactionQty'], 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 1, Seasonality))
                sarimax_result = sarimax_model.fit()

                # Forecast with SARIMAX
                if len(test_data) > 0:
                    sarimax_forecast = sarimax_result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels').rename('SARIMAX')
                    accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], sarimax_forecast)
                else:
                    accuracy, mae = 0, 0
                        
                sarimax_model_s = SARIMAX(resampled_df['TransactionQty'], 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 1, Seasonality))
                sarimax_result_s = sarimax_model_s.fit()

                # Forecast with SARIMAX
                sarimax_forecast_s = sarimax_result_s.predict(start=len(resampled_df), 
                                              end=len(resampled_df) + Horizon - 1, 
                                              typ='levels').rename('SARIMAX')

                # Generate forecast dates
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)

                best_model_name = "SARIMAX"
                best_forecast = sarimax_forecast_s

                forecast_dict = [
                    {
                        "Date": date.strftime('%Y-%m-%d'),
                        "TransactionQty": max(0, int(value)),  # Ensure non-negative values
                        "Sigma": 0
                    }
                    for date, value in zip(forecast_dates, best_forecast)
                ]

                output_dict = {
                    "Forecast": forecast_dict,
                    "BestModel": [
                        {
                            "Best Model": best_model_name,
                            "Accuracy": accuracy,
                            "MAE": mae,
                            "Parameter1": 0,
                            "Parameter2": 0,
                            "Parameter3": 0,
                            "Parameter4": 0,
                            "Parameter5": 0,
                            "Parameter6": 0
                        }
                    ]
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"SARIMAX Model Error: {str(e)}"}

        elif Model == "ETS":
            try:
                # Check if we have enough data for ETS
                if len(train_data) < max(Seasonality, 10):
                    return {"error": f"Insufficient data for ETS model. Need at least {max(Seasonality, 10)} data points."}
                
                ets_model = ExponentialSmoothing(train_data['TransactionQty'], 
                                            trend='add', 
                                            seasonal='add', 
                                            seasonal_periods=Seasonality)
                ets_result = ets_model.fit()

                # Forecast with ETS
                if len(test_data) > 0:
                    ets_forecast = ets_result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1).rename('ETS')
                    accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], ets_forecast)
                else:
                    accuracy, mae = 0, 0
                        
                ets_model_s = ExponentialSmoothing(resampled_df['TransactionQty'], 
                                            trend='add', 
                                            seasonal='add', 
                                            seasonal_periods=Seasonality)
                ets_result_s = ets_model_s.fit()

                # Forecast with ETS
                ets_forecast_s = ets_result_s.predict(start=len(resampled_df), 
                                       end=len(resampled_df) + Horizon - 1).rename('ETS')

                # Generate forecast dates
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)

                best_model_name = "ETS"
                best_forecast = ets_forecast_s

                forecast_dict = [
                    {
                        "Date": date.strftime('%Y-%m-%d'),
                        "TransactionQty": max(0, int(value)),  # Ensure non-negative values
                        "Sigma": 0
                    }
                    for date, value in zip(forecast_dates, best_forecast)
                ]

                output_dict = {
                    "Forecast": forecast_dict,
                    "BestModel": [
                        {
                            "Best Model": best_model_name,
                            "Accuracy": accuracy,
                            "MAE": mae,
                            "Parameter1": 0,
                            "Parameter2": 0,
                            "Parameter3": 0,
                            "Parameter4": 0,
                            "Parameter5": 0,
                            "Parameter6": 0
                        }
                    ]
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"ETS Model Error: {str(e)}"}

        elif Model == "STL + ARIMA":
            try:
                # Check if we have enough data for STL + ARIMA
                if len(train_data) < 10:
                    return {"error": "Insufficient data for STL + ARIMA model. Need at least 10 data points."}
                
                stlf = STLForecast(train_data['TransactionQty'], ARIMA, model_kwargs={"order": (1, 1, 0)})
                stlf_result = stlf.fit()
                
                if len(test_data) > 0:
                    stl_forecast = stlf_result.forecast(steps=len(test_data)).rename('STL Forecast')
                    accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], stl_forecast)
                else:
                    accuracy, mae = 0, 0
                        
                stlf_s = STLForecast(resampled_df['TransactionQty'], ARIMA, model_kwargs={"order": (1, 1, 0)})
                stlf_result_s = stlf_s.fit()
                stl_forecast_s = stlf_result_s.forecast(Horizon).rename('STL Forecast')

                # Generate forecast dates
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)

                best_model_name = "STL + ARIMA"
                best_forecast = stl_forecast_s

                forecast_dict = [
                    {
                        "Date": date.strftime('%Y-%m-%d'),
                        "TransactionQty": max(0, int(value)),  # Ensure non-negative values
                        "Sigma": 0
                    }
                    for date, value in zip(forecast_dates, best_forecast)
                ]

                output_dict = {
                    "Forecast": forecast_dict,
                    "BestModel": [
                        {
                            "Best Model": best_model_name,
                            "Accuracy": accuracy,
                            "MAE": mae,
                            "Parameter1": 0,
                            "Parameter2": 0,
                            "Parameter3": 0,
                            "Parameter4": 0,
                            "Parameter5": 0,
                            "Parameter6": 0
                        }
                    ]
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"STL + ARIMA Model Error: {str(e)}"}

        else:
            # Auto model selection
            try:
                models_to_try = []
                results = {}
                
                # AR Model with different lags
                lag_values = [1, 2, 3, 4, 5]
                for lag in lag_values:
                    try:
                        if len(train_data) > lag:
                            ar_model = AutoReg(train_data['TransactionQty'], lags=lag)
                            ar_results = ar_model.fit()
                            if len(test_data) > 0:
                                ar_forecast = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
                                accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], ar_forecast)
                                results[f'AR_{lag}'] = {'mae': mae, 'accuracy': accuracy, 'params': {'lag': lag}}
                    except:
                        continue
                
                # ES Model with different alpha values
                alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
                for alpha in alpha_values:
                    try:
                        model = sm.tsa.ExponentialSmoothing(train_data['TransactionQty'], trend=None, seasonal=None, initialization_method="estimated")
                        model_results = model.fit(smoothing_level=alpha)
                        if len(test_data) > 0:
                            forecast = model_results.forecast(steps=len(test_data))
                            accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], forecast)
                            results[f'ES_{alpha}'] = {'mae': mae, 'accuracy': accuracy, 'params': {'alpha': alpha}}
                    except:
                        continue

                # SARIMAX Model (if enough data)
                if len(train_data) >= max(Seasonality, 10):
                    try:
                        sarimax_model = SARIMAX(train_data['TransactionQty'], 
                                            order=(1, 1, 1), 
                                            seasonal_order=(1, 1, 1, Seasonality))
                        sarimax_result = sarimax_model.fit()
                        if len(test_data) > 0:
                            sarimax_forecast = sarimax_result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')
                            accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], sarimax_forecast)
                            results['SARIMAX'] = {'mae': mae, 'accuracy': accuracy, 'params': {}}
                    except:
                        pass

                # ETS Model (if enough data)
                if len(train_data) >= max(Seasonality, 10):
                    try:
                        ets_model = ExponentialSmoothing(train_data['TransactionQty'], 
                                                    trend='add', 
                                                    seasonal='add', 
                                                    seasonal_periods=Seasonality)
                        ets_result = ets_model.fit()
                        if len(test_data) > 0:
                            ets_forecast = ets_result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
                            accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], ets_forecast)
                            results['ETS'] = {'mae': mae, 'accuracy': accuracy, 'params': {}}
                    except:
                        pass

                # STL + ARIMA Model (if enough data)
                if len(train_data) >= 10:
                    try:
                        stlf = STLForecast(train_data['TransactionQty'], ARIMA, model_kwargs={"order": (1, 1, 0)})
                        stlf_result = stlf.fit()
                        if len(test_data) > 0:
                            stl_forecast = stlf_result.forecast(steps=len(test_data))
                            accuracy, mae = safe_accuracy_calculation(test_data['TransactionQty'], stl_forecast)
                            results['STL+ARIMA'] = {'mae': mae, 'accuracy': accuracy, 'params': {}}
                    except:
                        pass

                if not results:
                    return {"error": "No models could be fitted successfully. Please check your data."}

                # Find best model based on lowest MAE
                best_model_key = min(results.keys(), key=lambda x: results[x]['mae'])
                best_result = results[best_model_key]
                
                # Generate forecast using the best model
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)
                
                # Refit the best model on full data and generate forecast
                if 'AR_' in best_model_key:
                    lag = best_result['params']['lag']
                    ar_model_s = AutoReg(resampled_df['TransactionQty'], lags=lag)
                    ar_results_s = ar_model_s.fit()
                    best_forecast = ar_results_s.predict(start=len(resampled_df), end=len(resampled_df) + Horizon - 1, dynamic=False)
                    best_model_name = "AR"
                elif 'ES_' in best_model_key:
                    alpha = best_result['params']['alpha']
                    model_s = sm.tsa.ExponentialSmoothing(resampled_df['TransactionQty'], trend=None, seasonal=None, initialization_method="estimated")
                    model_results_s = model_s.fit(smoothing_level=alpha)
                    best_forecast = model_results_s.forecast(steps=Horizon)
                    best_model_name = "ES"
                elif best_model_key == 'SARIMAX':
                    sarimax_model_s = SARIMAX(resampled_df['TransactionQty'], 
                                        order=(1, 1, 1), 
                                        seasonal_order=(1, 1, 1, Seasonality))
                    sarimax_result_s = sarimax_model_s.fit()
                    best_forecast = sarimax_result_s.predict(start=len(resampled_df), end=len(resampled_df) + Horizon - 1, typ='levels')
                    best_model_name = "SARIMAX"
                elif best_model_key == 'ETS':
                    ets_model_s = ExponentialSmoothing(resampled_df['TransactionQty'], 
                                                trend='add', 
                                                seasonal='add', 
                                                seasonal_periods=Seasonality)
                    ets_result_s = ets_model_s.fit()
                    best_forecast = ets_result_s.predict(start=len(resampled_df), end=len(resampled_df) + Horizon - 1)
                    best_model_name = "ETS"
                elif best_model_key == 'STL+ARIMA':
                    stlf_s = STLForecast(resampled_df['TransactionQty'], ARIMA, model_kwargs={"order": (1, 1, 0)})
                    stlf_result_s = stlf_s.fit()
                    best_forecast = stlf_result_s.forecast(Horizon)
                    best_model_name = "STL + ARIMA"

                forecast_dict = [
                    {
                        "Date": date.strftime('%Y-%m-%d'),
                        "TransactionQty": max(0, int(value)),  # Ensure non-negative values
                        "Sigma": 0
                    }
                    for date, value in zip(forecast_dates, best_forecast)
                ]

                output_dict = {
                    "Forecast": forecast_dict,
                    "BestModel": [
                        {
                            "Best Model": best_model_name,
                            "Accuracy": best_result['accuracy'],
                            "MAE": best_result['mae'],
                            "Parameter1": 0,
                            "Parameter2": 0,
                            "Parameter3": 0,
                            "Parameter4": 0,
                            "Parameter5": 0,
                            "Parameter6": 0
                        }
                    ]
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"Auto Model Selection Error: {str(e)}"}

    except Exception as e:
        error = str(e)
        return {"error": error}