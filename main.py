import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons
from Read import *
from ImproveBorderline import *
from ImproveBorderline2 import *
from BA_SMOTE import *
from Classifier import *
from Sampling import *
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

data = shuttle_2_vs_5

X = data.iloc[:, data.columns != "Class"]
y = data.iloc[:, data.columns == "Class"]
# X, y = make_circles(n_samples=(650, 200), factor=0.2, noise=0.2, random_state=42)
# X, y = make_moons(n_samples=(650, 200), noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# origin_dt_scores = DecisionTree("Origin", X_train, X_test, y_train, y_test)
# origin_rf_scores = RandomForest("Origin", X_train, X_test, y_train, y_test)
# origin_svm_scores = SVM("Origin", X_train, X_test, y_train, y_test)
# origin_nb_scores = NB("Origin", X_train, X_test, y_train, y_test)
# origin_knn_scores = KNN("Origin", X_train, X_test, y_train, y_test)
# origin_lr_scores = LR("Origin", X_train, X_test, y_train, y_test)
# origin_ada_scores = Adaboost("Origin", X_train, X_test, y_train, y_test)
# # origin_light_scores = Light("Origin", X_train, X_test, y_train, y_test)
# # origin_xgb_scores = XGB("Origin", X_train, X_test, y_train, y_test)
#
# # SMOTE
# X_smote, y_smote = smote(X_train, y_train)
#
# smote_dt_scores = DecisionTree("SMOTE", X_smote, X_test, y_smote, y_test)
# smote_rf_scores = RandomForest("SMOTE", X_smote, X_test, y_smote, y_test)
# smote_svm_scores = SVM("SMOTE", X_smote, X_test, y_smote, y_test)
# smote_nb_scores = NB("SMOTE", X_smote, X_test, y_smote, y_test)
# smote_knn_scores = KNN("SMOTE", X_smote, X_test, y_smote, y_test)
# smote_lr_scores = LR("SMOTE", X_smote, X_test, y_smote, y_test)
# smote_ada_scores = Adaboost("SMOTE", X_smote, X_test, y_smote, y_test)
# # smote_light_scores = Light("SMOTE", X_smote, X_test, y_smote, y_test)
# # smote_xgb_scores = XGB("SMOTE", X_smote, X_test, y_smote, y_test)
#
# # BorderlineSMOTE
# X_borderline, y_borderline = borderline(X_train, y_train)
#
# borderline_dt_scores = DecisionTree("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# borderline_rf_scores = RandomForest("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# borderline_svm_scores = SVM("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# borderline_nb_scores = NB("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# borderline_knn_scores = KNN("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# borderline_lr_scores = LR("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# borderline_ada_scores = Adaboost("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# # borderline_light_scores = Light("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
# # borderline_xgb_scores = XGB("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
#
# # BorderlineSMOTE2
# X_borderline2, y_borderline2 = borderline2(X_train, y_train)
#
# borderline2_dt_scores = DecisionTree("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# borderline2_rf_scores = RandomForest("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# borderline2_svm_scores = SVM("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# borderline2_nb_scores = NB("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# borderline2_knn_scores = KNN("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# borderline2_lr_scores = LR("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# borderline2_ada_scores = Adaboost("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# # borderline2_light_scores = Light("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
# # borderline2_xgb_scores = XGB("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
#
# # ADASYN
# X_adasyn, y_adasyn = adasyn(X_train, y_train)
#
# adasyn_dt_scores = DecisionTree("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# adasyn_rf_scores = RandomForest("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# adasyn_svm_scores = SVM("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# adasyn_nb_scores = NB("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# adasyn_knn_scores = KNN("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# adasyn_lr_scores = LR("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# adasyn_ada_scores = Adaboost("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# # adasyn_light_scores = Light("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
# # adasyn_xgb_scores = XGB("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
#
# # KMeansSMOTE
# # X_kmsmote, y_kmsmote = kmsmote(X_train, y_train)
# #
# # kmsmote_dt_scores = DecisionTree("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
# # kmsmote_rf_scores = RandomForest("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
# # kmsmote_svm_scores = SVM("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
# # kmsmote_nb_scores = NB("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
# # kmsmote_knn_scores = KNN("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
#
# # OneSidedSelection
# # X_oss, y_oss = onesidedselection(X_train, y_train)
# #
# # oss_dt_scores = DecisionTree("OneSidedSelection", X_oss, X_test, y_oss, y_test)
# # oss_rf_scores = RandomForest("OneSidedSelection", X_oss, X_test, y_oss, y_test)
# # oss_svm_scores = SVM("OneSidedSelection", X_oss, X_test, y_oss, y_test)
# # oss_nb_scores = NB("OneSidedSelection", X_oss, X_test, y_oss, y_test)
# # oss_knn_scores = KNN("OneSidedSelection", X_oss, X_test, y_oss, y_test)
# #
# # # TomekLinks
# # X_tomeklinks, y_tomeklinks = tomeklinks(X_train, y_train)
# #
# # tomeklinks_dt_scores = DecisionTree("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_rf_scores = RandomForest("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_svm_scores = SVM("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_nb_scores = NB("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_knn_scores = KNN("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_lr_scores = LR("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_ada_scores = Adaboost("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # # tomeklinks_light_scores = Light("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # # tomeklinks_xgb_scores = XGB("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# #
# # SMOTETomek
# X_smotetomek, y_smotetomek = smotetomek(X_train, y_train)
#
# smotetomek_dt_scores = DecisionTree("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# smotetomek_rf_scores = RandomForest("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# smotetomek_svm_scores = SVM("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# smotetomek_nb_scores = NB("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# smotetomek_knn_scores = KNN("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# smotetomek_lr_scores = LR("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# smotetomek_ada_scores = Adaboost("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# # smotetomek_light_scores = Light("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
# # smotetomek_xgb_scores = XGB("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
#
# # SMOTE_IPF
# X_train_ipf = np.array(X_train)
# y_train_ipf = np.array(y_train).flatten()
# X_smote_ipf, y_smote_ipf = smote_ipf(X_train_ipf, y_train_ipf)
#
# smote_ipf_dt_scores = DecisionTree("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# smote_ipf_rf_scores = RandomForest("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# smote_ipf_svm_scores = SVM("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# smote_ipf_nb_scores = NB("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# smote_ipf_knn_scores = KNN("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# smote_ipf_lr_scores = LR("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# smote_ipf_ada_scores = Adaboost("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# # smote_ipf_light_scores = Light("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
# # smote_ipf_xgb_scores = XGB("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
#
# # RSMOTE_V8
# X_train_rsmote = np.array(X_train)
# y_train_rsmote = np.array(y_train).flatten()
# X_rsmote, y_rsmote = rsmote(X_train_rsmote, y_train_rsmote)
#
# rsmote_dt_scores = DecisionTree("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# rsmote_rf_scores = RandomForest("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# rsmote_svm_scores = SVM("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# rsmote_nb_scores = NB("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# rsmote_knn_scores = KNN("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# rsmote_lr_scores = LR("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# rsmote_ada_scores = Adaboost("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# # rsmote_light_scores = Light("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
# # rsmote_xgb_scores = XGB("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)

# ImproveBorderline
train = pd.concat([X_train, y_train], axis=1)
# train = pd.concat([pd.DataFrame(X_train, columns=['Z1', 'Z2']), pd.DataFrame(y_train, columns=['Class'])], axis=1)

synthetic = ImproveBorderline(train)
# print(f"synthetic.shape = {synthetic.shape}")

X_improve = np.r_[X_train, synthetic]
y_improve = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic.shape[0]))]
# X_improve = np.r_[pd.DataFrame(X_train, columns=['Z1', 'Z2']), synthetic]
# y_improve = np.r_[np.array(pd.DataFrame(y_train, columns=['Class']).values).flatten(), np.ones((synthetic.shape[0]))]

# improve_dt_scores = DecisionTree("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_rf_scores = RandomForest("ImproveBorderline", X_improve, X_test, y_improve, y_test)
improve_svm_scores = SVM("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_nb_scores = NB("ImproveBorderline", X_improve, X_test, y_improve, y_test)
improve_knn_scores = KNN("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_lr_scores = LR("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_ada_scores = Adaboost("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_light_scores = Light("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_xgb_scores = XGB("ImproveBorderline", X_improve, X_test, y_improve, y_test)

# ImproveBorderline2
# synthetic2 = ImproveBorderline2(train)
# # print(f"synthetic2.shape = {synthetic2.shape}")
#
# X_improve2 = np.r_[X_train, synthetic2]
# y_improve2 = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic2.shape[0]))]
#
# improve2_dt_scores = DecisionTree("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_rf_scores = RandomForest("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_svm_scores = SVM("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_nb_scores = NB("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_knn_scores = KNN("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_lr_scores = LR("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_ada_scores = Adaboost("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_light_scores = Light("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# improve2_xgb_scores = XGB("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)

# BA_SMOTE
synthetic3 = BA_SMOTE(train)
# print(f"synthetic3.shape = {synthetic3.shape}")
# print(f"synthetic3 = {synthetic3}")

# if len(synthetic3 != 0):
X_ba_smote = np.r_[X_train, synthetic3]
y_ba_smote = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic3.shape[0]))]
# X_ba_smote = np.r_[pd.DataFrame(X_train, columns=['Z1', 'Z2']), synthetic3]
# y_ba_smote = np.r_[np.array(pd.DataFrame(y_train, columns=['Class']).values).flatten(), np.ones((synthetic3.shape[0]))]
# print(X_ba_smote)
# print(y_ba_smote)

# ba_smote_dt_scores = DecisionTree("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
# ba_smote_rf_scores = RandomForest("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
ba_smote_svm_scores = SVM("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
# ba_smote_nb_scores = NB("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
ba_smote_knn_scores = KNN("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
# ba_smote_lr_scores = LR("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
# ba_smote_ada_scores = Adaboost("BA_SMOTE", X_ba_smote, X_test, y_ba_smote, y_test)
