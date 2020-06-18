import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import warnings


def resample_data(data, freq, cols=None, time_col = None, method=np.mean):
    """
    function to resample timeseries data to given frequency
    :params
        data - DataFrame or DataFrameGroupBy to be resampled
        freq - target frequency to resample
        cols - if multiple columns in the dataframe, columns for which the resample will apply
        time_col - if DF does not have a DatetimeIndex or similar, column to reindex
        method - function to apply in the resample
    :results
        - DataFrame with resampled data
    """
    if time_col is None:
        if cols is None: func_data = data
        else: func_data = data[cols]
        return func_data.resample(freq).apply(method)
    else:
        if cols is None: func_data = data
        else: func_data = data[cols+[time_col]]
        return func_data.resample(freq, on=time_col).apply(method)
    
def plot_seasonal_decompose(data, y, freq, time_col=None, figsize=(16,12)):
    """
    Function to plot the seasonal decomposition of a time series
    :params
        data - DataFrame with Timeseries
        y - name of the column to decompose
        freq - int, frequency of seasonality to be analyzed
        time_col - if DF does not have a DatetimeIndex or similar, column to reindex
        figsize - size of the figure to plot
    """
    if time_col is not None: data=data.set_index(time_col)
    # get decomposition of timeseries
    decomposition = seasonal_decompose(data[y], freq=freq)
    # Plot decomposition
    fig = decomposition.plot()
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    fig.suptitle("Seasonal Decomposition of the Series", fontsize=16)
    plt.xlabel("Time")
    plt.show()

def plot_acf_pacf(data, y, figsize=(20,10), acf_lags=10, pacf_lags=10):
    """
    Function to plot autocorrelation and partiol autocorrelation of timeseries
    :params
        data - DataFrame with timeseries
        y - name of the column to analyze
        figsize - size of the figure to plot
        acf_lags - number of lags to plot on autocorrelation graph
        pacf_lags - number of lags to plot on partial autocorrelation graph
    """
    
    to_model = data[y]
    fig,ax = plt.subplots(2,1,figsize=figsize)
    # Plot ACF and PACF for time series using 30 lags maximum
    fig = plot_acf(to_model.dropna(), lags=acf_lags, ax=ax[0])
    fig = plot_pacf(to_model.dropna(), lags=pacf_lags, ax=ax[1])
    plt.suptitle("Autocorrelation and Partial Correlation for the series", fontsize=14)
    plt.xlabel("Lag Number")
    plt.show()
    
def param_grid_product(param_grid):
    """
    Function to make combinations of parameter grid input
    Useful for cross validation purposes
    :params
        param_grid - dictionary with name of the parameter and list of options
    :return
        list of possible combinations using dictionaries to store parameters
    """
    return [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

def ts_crossvalidation(X, param_grid, y=None, cv=5, model="ARIMA", ignore_warnings=True):
    """
    Function to perform Timeseries crossv validation using nested cross validation
    Only possible to use ARIMA and SVR models right now
    :params
        X - Prediction variable with time series
        param_grid - dictionary with name of the parameter and list of options
        y - Target variable for SVR
        cv - number of folds to use in each validation
        model - model on which cross validation will be performed
        ignore_warnings - Option to silence warnings
    """
    if ignore_warnings:
        # Ignore all warnigs for Nested Cross Validations
        warnings.filterwarnings("ignore")
    # Time series cross validation initialization
    tscv = TimeSeriesSplit(n_splits=cv)
    # Initialization of results
    grid_search_result = pd.DataFrame()
    # Getting possible combinations of parameters
    grid_list = param_grid_product(param_grid)
    if model == "ARIMA":
        # -----
        # ARIMA Model Cross validation start
        # -----
        assert 'order' in param_grid, "No order parameters for ARIMA"
        # Loop over possible combinations of parameters
        for grid_dict in grid_list:
            order = grid_dict['order']
            # Set defaults for ARIMA parameters
            param_arima = {
                'seasonal_order':(0,0,0,0),
                'freq':None,
                'enforce_stationarity':True,
                'enforce_invertibility':True
            }
            # Update ARIMA parameters if they were specified
            for key in param_arima:
                if key in grid_dict:
                    param_arima[key] = grid_dict[key]
            # Initialize error lists
            rmse_list = []
            aic_list = []
            # Nested cross validation run per each configuration
            for train_index, test_index in tscv.split(X):
                # Train and test initialization for specific nested crossvalidation step
                X_train, X_test = X[train_index], X[test_index]
                # Model initialization and training
                try:
                    model = SARIMAX(X_train, 
                                    freq=param_arima['freq'], 
                                    order=order, 
                                    seasonal_order=param_arima['seasonal_order'],
                                    enforce_stationarity=param_arima['enforce_stationarity'], 
                                    enforce_invertibility=param_arima['enforce_invertibility']).fit(disp=0)
                    # Model test for crossvalidation step
                    pred = model.predict(X_test.index[0],X_test.index[-1])
                    error = np.sqrt(mean_squared_error(X_test, pred))
                    # Save results for crossvalidation step
                    rmse_list.append(error)
                    aic_list.append(model.aic)
                except:
                    # If error continue to next model evaluation
                    continue
            # Consolidate metrics for parameter configuration using mean
            try:
                total_error = np.mean(rmse_list)
                total_aic = np.mean(aic_list)
            except:
                # If error continue to next parameter configuration
                continue
            # Save results on main DataFrame
            to_append = pd.DataFrame([{'name':'ARIMA{}x{}'.format(order, param_arima['seasonal_order']),'AIC':total_aic, 'RMSE':total_error}])
            grid_search_result = grid_search_result.append(to_append, sort=False, ignore_index=True)
                
        
    elif model == "SVR":
        # -----
        # SVR Model Cross validation start
        # -----
        assert y is not None, "No target variable samples (y) for SVR"
        for grid_dict in grid_list:
            param_svr= {
                'C':1,
                'kernel':'rbf',
                'gamma':'scale',
            }
            for key in param_svr:
                if key in grid_dict:
                    param_svr[key] = grid_dict[key]
            # Initialize confidence lists and model
            confidence = None
            conf_list = []
            svr_rbf = SVR(kernel=param_svr['kernel'], C=param_svr['C'], gamma=param_svr['gamma']) 
            for train_index, test_index in tscv.split(X):
                # Train and test initialization for specific nested crossvalidation step
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                try:
                    # Model training 
                    svr_rbf.fit(X_train, y_train)
                    # Model test for nested crossvalidation step
                    svm_confidence = svr_rbf.score(X_test, y_test)
                    # Accuracy must be a valid number, otherwise the model failed to converge
                    assert( 0 <= svm_confidence <= 1 )
                    conf_list.append(svm_confidence)
                except: continue
            # Consolidate confidence for parameter configuration using mean
            try: confidence = np.mean(conf_list) 
            except: continue
            # Save results on main DataFrame
            to_append = pd.DataFrame([{'kernel':param_svr['kernel'],'gamma':param_svr['gamma'], 'C':param_svr['C'], 'confidence':confidence}])
            grid_search_result = grid_search_result.append(to_append, sort=False, ignore_index=True)
    
    return grid_search_result

def ARIMA_prediction(start, end, model, conf_int = True):
    """
    Function to get ARIMA predictions and interval confidence
    :params 
        start - Datetime object of when I want to start predicting
        end - Datetime object of when I want to finish predicting
        model - trained ARIMA model
        conf_int - bool, whether or not to get confidence intervals
    :return
        pred - Series with predictions
        pred_ci - Predicted confidence intervals
    """
    # Initialize return variables
    pred = None
    pred_ci = None
    
    prediction = model.get_prediction(start=start, end=end)
    pred = prediction.predicted_mean
    
    if conf_int: 
        pred_ci = prediction.conf_int()
    
    return pred, pred_ci

def ARIMA_plot_prediction(start, end, model, data=None, conf_int = True, figsize=(14,7)):
    """
    Function to plot ARIMA predictions and interval confidence
    :params 
        start - Datetime object of when I want to start predicting
        end - Datetime object of when I want to finish predicting
        model - trained ARIMA model
        data - if available test samples
        conf_int - bool, whether or not to get confidence intervals
    :return
        pred - Series with predictions
        pred_ci - Predicted confidence intervals
    """
    # Get predictions and confidence intervals
    pred, pred_ci = ARIMA_prediction(start, end, model, conf_int)
    # Plot predictions
    ax = pred.plot(label='prediction', figsize=figsize)
    #Plot observed data
    if data is not None:
        data.plot(ax = ax, label='observed', alpha=.7)
    # Plot confidence interval
    if conf_int: 
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        
    plt.title("ARIMA predictions")
    plt.legend()
    plt.show()