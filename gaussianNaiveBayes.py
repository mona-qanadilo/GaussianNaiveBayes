

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nClass Distribution:")
print(df['species'].value_counts())


X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class GaussianNBFromScratch:
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.class_priors_ = np.zeros(n_classes)
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = X_c.shape[0] / float(n_samples)
            self.mean_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X.values]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []


        for idx, c in enumerate(self.classes_):
            prior = np.log(self.class_priors_[idx])
            posterior = np.sum(np.log(self._gaussian_pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)


        return self.classes_[np.argmax(posteriors)]

    def _gaussian_pdf(self, class_idx, x):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


gnb_scratch = GaussianNBFromScratch()
gnb_scratch.fit(X_train, y_train)
y_pred_scratch = gnb_scratch.predict(X_test)

gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train, y_train)
y_pred_sklearn = gnb_sklearn.predict(X_test)

print("Scratch implementation results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_scratch):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_scratch))

print("Scikit learn implementation results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_sklearn))

custom_input = np.array([[4.9, 3.0, 1.4, 0.2]])

custom_df = pd.DataFrame(custom_input, columns=X_train.columns)

scratch_pred = gnb_scratch.predict(custom_df)
sklearn_pred = gnb_sklearn.predict(custom_input)

print("From Scratch gnb")
print(iris.target_names[scratch_pred][0])
print("Scikit learn")
print(iris.target_names[sklearn_pred][0])



