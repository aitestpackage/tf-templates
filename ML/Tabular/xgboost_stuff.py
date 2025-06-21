# !pip install xgboost
from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
bst = XGBClassifier(
    n_estimators=2,
    max_depth=2, 
    max_leaves=1000,
    learning_rate=1, 
    eta=0.3, # step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
    gamma=0.01,
    alpha=1, # L1 regularization term on weights
    grow_policy="depthwise", # lossguide
    tree_method="hist", 
    objective='multi:softprob',  # Multiclass classification
    eval_metric='mlogloss',      # Evaluation metric for multiclass classification
    device="cuda",
    verbosity=1, # 0 to 3
)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)

# objective [ default=reg:linear ]
# specify the learning task and the corresponding learning objective, and the objective options are below:
# “reg:linear” –linear regression
# “reg:logistic” –logistic regression
# “binary:logistic” –logistic regression for binary classification, output probability
# “binary:logitraw” –logistic regression for binary classification, output score before logistic transformation
# “count:poisson” –poisson regression for count data, output mean of poisson distribution
# max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
# “multi:softmax” –set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
# “multi:softprob” –same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
# “rank:pairwise” –set XGBoost to do ranking task by minimizing the pairwise loss
# base_score [ default=0.5 ]
# the initial prediction score of all instances, global bias
# eval_metric [ default according to objective ]
# evaluation metrics for validation data, a default metric will be assigned according to objective( rmse for regression, and error for classification, mean average precision for ranking )
# User can add multiple evaluation metrics, for python user, remember to pass the metrics in as list of parameters pairs instead of map, so that latter ‘eval_metric’ won’t override previous one
# The choices are listed below:
# “rmse”: root mean square error
# “logloss”: negative log-likelihood
# “error”: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
# “merror”: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
# “mlogloss”: Multiclass logloss
# “auc”: Area under the curve for ranking evaluation.
# “ndcg”:Normalized Discounted Cumulative Gain
# “map”:Mean average precision
# “ndcg@n”,”map@n”: n can be assigned as an integer to cut off the top positions in the lists for evaluation.
# “ndcg-”,”map-”,”ndcg@n-”,”map@n-”: In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding “-” in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions. training repeatively
# seed [ default=0 ]