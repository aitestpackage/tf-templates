# !pip install sktime
from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA
y = load_airline()
forecaster = ARIMA(  
    order=(1, 1, 0),
    seasonal_order=(0, 1, 0, 12),
    suppress_warnings=True)
forecaster.fit(y)  
y_pred = forecaster.predict(fh=[1,2,3])  