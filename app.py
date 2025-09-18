import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Import the forecasting libraries
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.stl import STLForecast
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

# Complete implementation of the original forecasting function
def run_forecast(data):
    """
    Complete implementation matching the original score_3.py logic
    """
    try:
        # Load the input data (JSON format)
        input_data = json.loads(data) if isinstance(data, str) else data

        # Convert the input data to a DataFrame
        web_service_input_data = input_data['Inputs']['web_service_input']

        Horizon = input_data['Inputs']['input1'][0]['horizon']
        Seasonality = input_data['Inputs']['input1'][0]['seasonality']
        grouping_parameter = input_data['Inputs']['input1'][0]['grouping_parameter']
        Model = input_data['Inputs']['input1'][0]['Model']
        parameter1 = input_data['Inputs']['input1'][0]['parameter1']

        # Convert the list of dictionaries to a DataFrame
        filtered_df1 = pd.DataFrame(web_service_input_data)
        
        filtered_df1['DateKey'] = pd.to_datetime(filtered_df1['DateKey'])

        # Create separate columns for Year, MonthNumber, QuarterNumber, and WeekNumber
        filtered_df1['Year'] = filtered_df1['DateKey'].dt.year
        filtered_df1['MonthNumber'] = filtered_df1['DateKey'].dt.month
        filtered_df1['QuarterNumber'] = filtered_df1['DateKey'].dt.quarter
        filtered_df1['WeekNumber'] = filtered_df1['DateKey'].dt.isocalendar().week
        filtered_df1['DayNumber'] = filtered_df1['DateKey'].dt.day

        # Create a new column for the selected parameter
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

        # Preprocess the data
        summary_df['parameter'] = pd.to_datetime(summary_df['parameter'])
        filtered_df2 = summary_df[['parameter', 'TransactionQty']]
        filtered_df2.set_index('parameter', inplace=True)

        # Resample the data based on the grouping parameter
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
            raise ValueError("Invalid grouping parameter.")

        # Replace NaN values and round
        resampled_df['TransactionQty'] = resampled_df['TransactionQty'].fillna(0).round(2)

        # Check if we have enough data points
        if len(resampled_df) < 2:
            return {"error": "Insufficient data points for forecasting. Need at least 2 data points."}

        # Split data
        train_end = max(1, int(len(resampled_df) * 0.8))
        train_data = resampled_df.iloc[:train_end]
        test_data = resampled_df.iloc[train_end:] if len(resampled_df) > train_end else resampled_df.iloc[-1:]

        # Helper functions
        def safe_accuracy_calculation(actual, predicted):
            if len(actual) == 0 or len(predicted) == 0:
                return 0, 0
            
            mae = mean_absolute_error(actual, predicted)
            mean_actual = np.mean(actual)
            
            if mean_actual == 0 or np.isnan(mean_actual) or np.isinf(mean_actual):
                return 0, mae
            
            accuracy = max(0, 100 - abs((mae / mean_actual) * 100))
            return int(accuracy), int(mae)

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

        # Model implementations - COMPLETE ORIGINAL LOGIC
        if Model == "AR":
            try:
                ar_model = AutoReg(resampled_df['TransactionQty'], lags=parameter1)
                ar_results = ar_model.fit()
                
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)
                ar_forecast = ar_results.predict(start=len(resampled_df), end=len(resampled_df) + Horizon - 1, dynamic=False)
                
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
                        "TransactionQty": max(0, int(value)),
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
                    ],
                    "HistoricalData": resampled_df.reset_index().to_dict('records')
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"AR Model Error: {str(e)}"}

        elif Model == "ES":
            try:
                model_s = sm.tsa.ExponentialSmoothing(resampled_df['TransactionQty'], trend=None, seasonal=None, initialization_method="estimated")
                model_results_s = model_s.fit(smoothing_level=parameter1)
                
                forecast_dates = generate_forecast_dates(resampled_df.index[-1], Horizon, grouping_parameter)
                forecast_s = model_results_s.forecast(steps=Horizon)
                
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
                        "TransactionQty": max(0, int(value)),
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
                    ],
                    "HistoricalData": resampled_df.reset_index().to_dict('records')
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
                        "TransactionQty": max(0, int(value)),
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
                    ],
                    "HistoricalData": resampled_df.reset_index().to_dict('records')
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
                        "TransactionQty": max(0, int(value)),
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
                    ],
                    "HistoricalData": resampled_df.reset_index().to_dict('records')
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
                        "TransactionQty": max(0, int(value)),
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
                    ],
                    "HistoricalData": resampled_df.reset_index().to_dict('records')
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"STL + ARIMA Model Error: {str(e)}"}

        else:
            # Auto model selection - COMPLETE IMPLEMENTATION
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
                        "TransactionQty": max(0, int(value)),
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
                    ],
                    "HistoricalData": resampled_df.reset_index().to_dict('records')
                }
                return output_dict
                
            except Exception as e:
                return {"error": f"Auto Model Selection Error: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
    
    st.title("📈 Sales Forecasting Dashboard")
    st.write("Upload your Excel file and configure forecasting parameters")

    # Sidebar for controls
    st.sidebar.header("📊 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.sidebar.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Display basic info about the dataset
            st.subheader("📋 Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Info:**")
                st.write(f"- Rows: {len(df)}")
                st.write(f"- Columns: {len(df.columns)}")
                st.write(f"- Column names: {list(df.columns)}")
                
            with col2:
                st.write("**Sample Data:**")
                st.dataframe(df.head())
            
            # Validate required columns
            required_columns = ['ItemNumber', 'Date', 'Quantity']
            if not all(col in df.columns for col in required_columns):
                st.error(f"Missing required columns. Expected: {required_columns}")
                return
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sidebar configuration
            st.sidebar.header("🔧 Forecasting Parameters")
            
            # Item selection
            unique_items = df['ItemNumber'].unique()
            selected_item = st.sidebar.selectbox("Select Item", unique_items)
            
            # Filter data for selected item
            filtered_df = df[df['ItemNumber'] == selected_item].copy()
            
            # Date range selection
            min_date = filtered_df['Date'].min()
            max_date = filtered_df['Date'].max()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['Date'] >= pd.to_datetime(start_date)) &
                    (filtered_df['Date'] <= pd.to_datetime(end_date))
                ]
            
            # Forecasting parameters
            grouping_parameter = st.sidebar.selectbox(
                "Grouping Period",
                options=['D', 'W', 'M', 'Q', 'Y'],
                format_func=lambda x: {
                    'D': 'Daily',
                    'W': 'Weekly', 
                    'M': 'Monthly',
                    'Q': 'Quarterly',
                    'Y': 'Yearly'
                }[x],
                index=2  # Default to Monthly
            )
            
            horizon = st.sidebar.number_input("Forecast Horizon", min_value=1, max_value=50, value=12)
            
            seasonality_options = {
                'D': 7,   # Weekly seasonality for daily data
                'W': 52,  # Yearly seasonality for weekly data
                'M': 12,  # Yearly seasonality for monthly data
                'Q': 4,   # Yearly seasonality for quarterly data
                'Y': 1    # No seasonality for yearly data
            }
            
            seasonality = st.sidebar.number_input(
                "Seasonality", 
                min_value=1, 
                max_value=52, 
                value=seasonality_options[grouping_parameter]
            )
            
            # Complete model options matching original implementation
            model_options = ["Auto", "AR", "ES", "SARIMAX", "ETS", "STL + ARIMA"]
            selected_model = st.sidebar.selectbox("Select Model", model_options)
            
            # Model parameters based on selected model
            parameter1 = 0.3  # Default value
            if selected_model == "AR":
                parameter1 = st.sidebar.number_input("AR Lag", min_value=1, max_value=10, value=1)
            elif selected_model == "ES":
                parameter1 = st.sidebar.slider("Smoothing Level (Alpha)", 0.1, 0.9, 0.3)
            elif selected_model == "SARIMAX":
                st.sidebar.info("SARIMAX uses automatic parameter selection")
            elif selected_model == "ETS":
                st.sidebar.info("ETS uses automatic parameter selection")
            elif selected_model == "STL + ARIMA":
                st.sidebar.info("STL + ARIMA uses automatic parameter selection")
            elif selected_model == "Auto":
                st.sidebar.info("Auto mode will test all models and select the best one")
            
            # Model information expander
            with st.sidebar.expander("ℹ️ Model Information"):
                st.write("**AR**: AutoRegressive model, good for data with trends")
                st.write("**ES**: Exponential Smoothing, simple and robust")
                st.write("**SARIMAX**: Seasonal ARIMA, handles seasonality and trends")
                st.write("**ETS**: Error, Trend, Seasonal decomposition")
                st.write("**STL + ARIMA**: Seasonal-Trend decomposition with ARIMA")
                st.write("**Auto**: Tests all models and picks the best performer")
            
            # Run forecasting button
            if st.sidebar.button("🚀 Run Forecast", type="primary"):
                with st.spinner("Running forecast analysis..."):
                    
                    # Prepare data in the format expected by the forecasting function
                    forecast_data = filtered_df[['Date', 'Quantity']].copy()
                    forecast_data = forecast_data.rename(columns={
                        'Date': 'DateKey',
                        'Quantity': 'TransactionQty'
                    })
                    
                    # Convert to the JSON format expected by the original function
                    json_data = {
                        "Inputs": {
                            "web_service_input": forecast_data.to_dict('records'),
                            "input1": [{
                                "horizon": horizon,
                                "seasonality": seasonality,
                                "grouping_parameter": grouping_parameter,
                                "Model": selected_model,
                                "parameter1": parameter1
                            }]
                        }
                    }
                    
                    # Run the forecast
                    result = run_forecast(json_data)
                    
                    if "error" in result:
                        st.error(f"Forecasting Error: {result['error']}")
                    else:
                        # Display results
                        st.subheader("🎯 Forecast Results")
                        
                        # Model performance metrics
                        best_model = result['BestModel'][0]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Best Model", best_model['Best Model'])
                        with col2:
                            st.metric("Accuracy", f"{best_model['Accuracy']}%")
                        with col3:
                            st.metric("MAE", f"{best_model['MAE']}")
                        with col4:
                            st.metric("Forecast Points", len(result['Forecast']))
                        
                        # Prepare data for visualization
                        historical_data = result.get('HistoricalData', [])
                        forecast_data_result = result['Forecast']
                        
                        # Create comprehensive visualization
                        if historical_data:
                            hist_df = pd.DataFrame(historical_data)
                            hist_df['parameter'] = pd.to_datetime(hist_df['parameter'])
                            hist_df['Type'] = 'Historical'
                            hist_df['Date'] = hist_df['parameter']
                            hist_df['Value'] = hist_df['TransactionQty']
                        else:
                            hist_df = pd.DataFrame()
                        
                        forecast_df = pd.DataFrame(forecast_data_result)
                        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                        forecast_df['Type'] = 'Forecast'
                        forecast_df['Value'] = forecast_df['TransactionQty']
                        
                        # Combine data for plotting
                        if not hist_df.empty:
                            plot_data = pd.concat([
                                hist_df[['Date', 'Value', 'Type']],
                                forecast_df[['Date', 'Value', 'Type']]
                            ], ignore_index=True)
                        else:
                            plot_data = forecast_df[['Date', 'Value', 'Type']]
                        
                        # Create enhanced plotly chart
                        fig = go.Figure()
                        
                        # Add historical data if available
                        if not hist_df.empty:
                            fig.add_trace(go.Scatter(
                                x=hist_df['Date'],
                                y=hist_df['Value'],
                                mode='lines+markers',
                                name='Historical Data',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=4)
                            ))
                        
                        # Add forecast data
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Value'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='#ff7f0e', width=3, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f'Sales Forecast for {selected_item} using {best_model["Best Model"]} Model',
                            xaxis_title='Date',
                            yaxis_title='Quantity',
                            hovermode='x unified',
                            showlegend=True,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display detailed results in tabs
                        tab1, tab2, tab3 = st.tabs(["📊 Forecast Table", "📈 Historical Data", "🔧 Model Details"])
                        
                        with tab1:
                            st.subheader("Forecast Results")
                            forecast_display = pd.DataFrame(forecast_data_result)
                            forecast_display['Date'] = pd.to_datetime(forecast_display['Date']).dt.strftime('%Y-%m-%d')
                            st.dataframe(forecast_display, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Forecast Mean", f"{forecast_display['TransactionQty'].mean():.2f}")
                            with col2:
                                st.metric("Forecast Total", f"{forecast_display['TransactionQty'].sum():.0f}")
                            with col3:
                                st.metric("Forecast Std", f"{forecast_display['TransactionQty'].std():.2f}")
                        
                        with tab2:
                            if historical_data:
                                st.subheader("Historical Data")
                                hist_display = pd.DataFrame(historical_data)
                                hist_display['parameter'] = pd.to_datetime(hist_display['parameter']).dt.strftime('%Y-%m-%d')
                                hist_display = hist_display.rename(columns={'parameter': 'Date'})
                                st.dataframe(hist_display, use_container_width=True)
                                
                                # Historical statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Historical Mean", f"{hist_display['TransactionQty'].mean():.2f}")
                                with col2:
                                    st.metric("Historical Total", f"{hist_display['TransactionQty'].sum():.0f}")
                                with col3:
                                    st.metric("Historical Std", f"{hist_display['TransactionQty'].std():.2f}")
                            else:
                                st.info("No historical data available for display")
                        
                        with tab3:
                            st.subheader("Model Performance Details")
                            model_info = pd.DataFrame([best_model])
                            st.dataframe(model_info, use_container_width=True)
                            
                            # Model explanation
                            model_name = best_model['Best Model']
                            if model_name == "AR":
                                st.write("**AutoRegressive (AR) Model**: Uses past values to predict future values. Good for data with trends.")
                            elif model_name == "ES":
                                st.write("**Exponential Smoothing (ES)**: Gives more weight to recent observations. Simple and robust for many time series.")
                            elif model_name == "SARIMAX":
                                st.write("**Seasonal ARIMA (SARIMAX)**: Handles both trend and seasonal patterns. Good for complex seasonal data.")
                            elif model_name == "ETS":
                                st.write("**Error, Trend, Seasonal (ETS)**: Decomposes the time series into error, trend, and seasonal components.")
                            elif model_name == "STL + ARIMA":
                                st.write("**STL + ARIMA**: Combines seasonal decomposition (STL) with ARIMA modeling for robust forecasting.")
                        
                        # Download options
                        st.subheader("📥 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download forecast results
                            csv_forecast = forecast_display.to_csv(index=False)
                            st.download_button(
                                label="Download Forecast Results",
                                data=csv_forecast,
                                file_name=f"forecast_{selected_item}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            if historical_data:
                                # Download historical data
                                csv_historical = hist_display.to_csv(index=False)
                                st.download_button(
                                    label="Download Historical Data",
                                    data=csv_historical,
                                    file_name=f"historical_{selected_item}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
            
            # Display current filtered data
            st.subheader(f"📊 Current Data for {selected_item}")
            st.write(f"Date range: {filtered_df['Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"Total records: {len(filtered_df)}")
            
            # Show data visualization
            if len(filtered_df) > 0:
                fig_current = px.line(filtered_df, x='Date', y='Quantity', 
                                    title=f'Historical Data for {selected_item}')
                fig_current.update_traces(line=dict(width=2))
                st.plotly_chart(fig_current, use_container_width=True)
                
                # Show statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Statistics:**")
                    st.write(f"- Mean: {filtered_df['Quantity'].mean():.2f}")
                    st.write(f"- Std: {filtered_df['Quantity'].std():.2f}")
                    st.write(f"- Min: {filtered_df['Quantity'].min()}")
                    st.write(f"- Max: {filtered_df['Quantity'].max()}")
                    st.write(f"- Total: {filtered_df['Quantity'].sum()}")
                
                with col2:
                    st.write("**Recent Data:**")
                    st.dataframe(filtered_df.tail(10))
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.write("Please ensure your Excel file has the correct format with columns: ItemNumber, Date, Quantity")
    
    else:
        st.info("👆 Please upload an Excel file to get started")
        
        # Show example of expected data format
        st.subheader("📝 Expected Data Format")
        example_data = {
            'ItemNumber': ['ITEM001', 'ITEM001', 'ITEM001', 'ITEM002', 'ITEM002'],
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02'],
            'Quantity': [23, 18, 27, 15, 20]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
        
        # Model comparison information
        st.subheader("🤖 Available Forecasting Models")
        
        models_info = {
            'Model': ['AR (AutoRegressive)', 'ES (Exponential Smoothing)', 'SARIMAX', 'ETS', 'STL + ARIMA', 'Auto Selection'],
            'Best For': [
                'Data with trends and patterns',
                'Simple, stable data',
                'Seasonal data with trends',
                'Complex seasonal patterns',
                'Data with strong seasonality',
                'Unknown data patterns'
            ],
            'Complexity': ['Medium', 'Low', 'High', 'High', 'High', 'Variable'],
            'Data Requirements': ['Medium', 'Low', 'High (seasonal)', 'High (seasonal)', 'Medium', 'Variable']
        }
        
        st.dataframe(pd.DataFrame(models_info), use_container_width=True)

if __name__ == "__main__":
    main()