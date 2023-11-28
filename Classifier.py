import numpy as np
from Metrics import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
np.random.seed(42)


def DecisionTree(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "DecisionTreeClassifier", "=" * 10)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("DT SCORES", scores)
    return scores


def RandomForest(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "RandomForestClassifier", "=" * 10)
    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("RF SCORES", scores)
    return scores


def SVM(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "SVM", "=" * 10)
    # model = SVC(probability=True, random_state=42)
    model = SVC(probability=True, kernel="linear", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("SVM SCORES", scores)
    return scores


def NB(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "GaussianNB", "=" * 10)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("GaussianNB SCORES", scores)
    return scores


def KNN(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "KNeighborsClassifier", "=" * 10)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("KNN SCORES", scores)
    return scores


def LR(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "LogisticRegression", "=" * 10)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("LR SCORES", scores)
    return scores


def Adaboost(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "AdaBoostClassifier", "=" * 10)
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("Adaboost SCORES", scores)
    return scores


def Light(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "LGBMClassifier", "=" * 10)
    model = LGBMClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("LightGBM SCORES", scores)
    return scores


def XGB(title, X_train, X_test, y_train, y_test):
    print("=" * 10, title, "XGBClassifier", "=" * 10)
    model = XGBClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_pro = model.predict_proba(X_test)[:, 1]
    scores = metric(y_test, y_pred, y_pred_pro)
    print("XGBoost SCORES", scores)
    return scores
