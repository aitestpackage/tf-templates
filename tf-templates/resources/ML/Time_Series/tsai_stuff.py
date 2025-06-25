# !pip install tsai
from tsai.data.preparation import *
ts = get_forecasting_time_series('Sunspots')
if ts is not None: # This is to prevent a test fail when the resources server is not available
    X, y = SlidingWindowSplitter(60, horizon=1)(ts)
    X, y = X.astype('float32'), y.astype('float32')
    splits = TSSplitter(235)(y)
    batch_tfms = [TSStandardize(by_var=True)]
    learn = TSForecaster(X, y, splits=splits, batch_tfms=batch_tfms, arch=None, arch_config=dict(fc_dropout=.5), metrics=mae, bs=512,
                         partial_n=.1, train_metrics=True, device=default_device())
    learn.fit_one_cycle(1)