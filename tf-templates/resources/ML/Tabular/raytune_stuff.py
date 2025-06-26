# !pip install xgboost > /dev/null
# !pip install "ray[data,train,tune,serve]" > /dev/null

from ray import tune
from xgboost import XGBClassifier

def train_model(config):
    clf = XGBClassifier(
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        eta=config["eta"],
        n_estimators=config['n_estimators'],
        enable_categorical=True,
        # device=""
    )
    # xgboost.predict_proba
    clf.fit(X_train, y_train)
    # result = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    result = np.sum(y_test == y_pred) / y_pred.shape[0]
    tune.report({"accuracy": result, "done": True})

config = {
    "max_depth": tune.randint(8, 24),
    "min_child_weight": tune.choice([1, 2, 3]),
    "eta": tune.loguniform(1e-4, 1e-1),
    "n_estimators": tune.randint(1000, 2000)
}
tuner = tune.Tuner(
    train_model,
    tune_config=tune.TuneConfig(num_samples=20),
    param_space=config,
)
results = tuner.fit()