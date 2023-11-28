from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.under_sampling import TomekLinks, OneSidedSelection
from imblearn.combine import SMOTETomek
from smote_variants import SMOTE_IPF, DBSMOTE, Gaussian_SMOTE
from RSMOTE_V8 import *


def smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def borderline(X, y):
    sm = BorderlineSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def borderline2(X, y):
    sm = BorderlineSMOTE(kind='borderline-2', random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def adasyn(X, y):
    sm = ADASYN(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def kmsmote(X, y):
    sm = KMeansSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def onesidedselection(X, y):
    oss = OneSidedSelection(random_state=42)
    X_res, y_res = oss.fit_resample(X, y)
    return X_res, y_res


def tomeklinks(X, y):
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)
    return X_res, y_res


def smotetomek(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res


def smote_ipf(X, y):
    ipf = SMOTE_IPF(random_state=42, n_neighbors=5)
    X_res, y_res = ipf.sample(X, y)
    return X_res, y_res


def rsmote(X, y):
    rs = RSmoteKClasses(random_state=42)
    X_res, y_res = rs.fit_resample(X, y)
    return X_res, y_res


def gaussian_smote(X, y):
    gaussian = Gaussian_SMOTE(random_state=42)
    X_res, y_res = gaussian.sample(X, y)
    return X_res, y_res

