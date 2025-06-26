# !pip install sktime
from sktime.classification.deep_learning import InceptionTimeClassifier
inception_clf = InceptionTimeClassifier(
    n_epochs=120,
    n_filters=12,
    kernel_size=24,
    depth=2,
    verbose=True
)  
inception_clf.fit(X_train, y_train)  