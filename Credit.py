import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from ImproveBorderline import *
from ImproveBorderline2 import *
from Sampling import *
from Classifier import *
import warnings
warnings.filterwarnings('ignore')

german = pd.read_csv('./data/credit/german.csv')
# print(german.shape)  # (1000, 21)

X = german.iloc[:, german.columns != "Class"]
y = german.iloc[:, german.columns == "Class"]

score_len = 3
origin_dt_scores_total = np.zeros(score_len)
origin_rf_scores_total = np.zeros(score_len)
origin_svm_scores_total = np.zeros(score_len)
origin_nb_scores_total = np.zeros(score_len)
origin_knn_scores_total = np.zeros(score_len)
origin_lr_scores_total = np.zeros(score_len)
origin_ada_scores_total = np.zeros(score_len)

smote_dt_scores_total = np.zeros(score_len)
smote_rf_scores_total = np.zeros(score_len)
smote_svm_scores_total = np.zeros(score_len)
smote_nb_scores_total = np.zeros(score_len)
smote_knn_scores_total = np.zeros(score_len)
smote_lr_scores_total = np.zeros(score_len)
smote_ada_scores_total = np.zeros(score_len)

borderline_dt_scores_total = np.zeros(score_len)
borderline_rf_scores_total = np.zeros(score_len)
borderline_svm_scores_total = np.zeros(score_len)
borderline_nb_scores_total = np.zeros(score_len)
borderline_knn_scores_total = np.zeros(score_len)
borderline_lr_scores_total = np.zeros(score_len)
borderline_ada_scores_total = np.zeros(score_len)

borderline2_dt_scores_total = np.zeros(score_len)
borderline2_rf_scores_total = np.zeros(score_len)
borderline2_svm_scores_total = np.zeros(score_len)
borderline2_nb_scores_total = np.zeros(score_len)
borderline2_knn_scores_total = np.zeros(score_len)
borderline2_lr_scores_total = np.zeros(score_len)
borderline2_ada_scores_total = np.zeros(score_len)

adasyn_dt_scores_total = np.zeros(score_len)
adasyn_rf_scores_total = np.zeros(score_len)
adasyn_svm_scores_total = np.zeros(score_len)
adasyn_nb_scores_total = np.zeros(score_len)
adasyn_knn_scores_total = np.zeros(score_len)
adasyn_lr_scores_total = np.zeros(score_len)
adasyn_ada_scores_total = np.zeros(score_len)

tomeklinks_dt_scores_total = np.zeros(score_len)
tomeklinks_rf_scores_total = np.zeros(score_len)
tomeklinks_svm_scores_total = np.zeros(score_len)
tomeklinks_nb_scores_total = np.zeros(score_len)
tomeklinks_knn_scores_total = np.zeros(score_len)
tomeklinks_lr_scores_total = np.zeros(score_len)
tomeklinks_ada_scores_total = np.zeros(score_len)

smotetomek_dt_scores_total = np.zeros(score_len)
smotetomek_rf_scores_total = np.zeros(score_len)
smotetomek_svm_scores_total = np.zeros(score_len)
smotetomek_nb_scores_total = np.zeros(score_len)
smotetomek_knn_scores_total = np.zeros(score_len)
smotetomek_lr_scores_total = np.zeros(score_len)
smotetomek_ada_scores_total = np.zeros(score_len)

improve_dt_scores_total = np.zeros(score_len)
improve_rf_scores_total = np.zeros(score_len)
improve_svm_scores_total = np.zeros(score_len)
improve_nb_scores_total = np.zeros(score_len)
improve_knn_scores_total = np.zeros(score_len)
improve_lr_scores_total = np.zeros(score_len)
improve_ada_scores_total = np.zeros(score_len)

improve2_dt_scores_total = np.zeros(score_len)
improve2_rf_scores_total = np.zeros(score_len)
improve2_svm_scores_total = np.zeros(score_len)
improve2_nb_scores_total = np.zeros(score_len)
improve2_knn_scores_total = np.zeros(score_len)
improve2_lr_scores_total = np.zeros(score_len)
improve2_ada_scores_total = np.zeros(score_len)

# 标准化
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

fold = 10
kf = KFold(n_splits=fold)
for train_index, test_index in kf.split(X):
    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_test = y.iloc[test_index, :]

    origin_dt_scores = DecisionTree("Origin", X_train, X_test, y_train, y_test)
    origin_dt_scores_total = origin_dt_scores_total + origin_dt_scores
    origin_rf_scores = RandomForest("Origin", X_train, X_test, y_train, y_test)
    origin_rf_scores_total = origin_rf_scores_total + origin_rf_scores
    origin_svm_scores = SVM("Origin", X_train, X_test, y_train, y_test)
    origin_svm_scores_total = origin_svm_scores_total + origin_svm_scores
    origin_nb_scores = NB("Origin", X_train, X_test, y_train, y_test)
    origin_nb_scores_total = origin_nb_scores_total + origin_nb_scores
    origin_knn_scores = KNN("Origin", X_train, X_test, y_train, y_test)
    origin_knn_scores_total = origin_knn_scores_total + origin_knn_scores
    origin_lr_scores = LR("Origin", X_train, X_test, y_train, y_test)
    origin_lr_scores_total = origin_lr_scores_total + origin_lr_scores
    origin_ada_scores = Adaboost("Origin", X_train, X_test, y_train, y_test)
    origin_ada_scores_total = origin_ada_scores_total + origin_ada_scores

    # SMOTE
    X_smote, y_smote = smote(X_train, y_train)

    smote_dt_scores = DecisionTree("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_dt_scores_total = smote_dt_scores_total + smote_dt_scores
    smote_rf_scores = RandomForest("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_rf_scores_total = smote_rf_scores_total + smote_rf_scores
    smote_svm_scores = SVM("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_svm_scores_total = smote_svm_scores_total + smote_svm_scores
    smote_nb_scores = NB("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_nb_scores_total = smote_nb_scores_total + smote_nb_scores
    smote_knn_scores = KNN("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_knn_scores_total = smote_knn_scores_total + smote_knn_scores
    smote_lr_scores = LR("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_lr_scores_total = smote_lr_scores_total + smote_lr_scores
    smote_ada_scores = Adaboost("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_ada_scores_total = smote_ada_scores_total + smote_ada_scores

    # BorderlineSMOTE
    X_borderline, y_borderline = borderline(X_train, y_train)

    borderline_dt_scores = DecisionTree("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_dt_scores_total = borderline_dt_scores_total + borderline_dt_scores
    borderline_rf_scores = RandomForest("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_rf_scores_total = borderline_rf_scores_total + borderline_rf_scores
    borderline_svm_scores = SVM("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_svm_scores_total = borderline_svm_scores_total + borderline_svm_scores
    borderline_nb_scores = NB("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_nb_scores_total = borderline_nb_scores_total + borderline_nb_scores
    borderline_knn_scores = KNN("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_knn_scores_total = borderline_knn_scores_total + borderline_knn_scores
    borderline_lr_scores = LR("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_lr_scores_total = borderline_lr_scores_total + borderline_lr_scores
    borderline_ada_scores = Adaboost("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_ada_scores_total = borderline_ada_scores_total + borderline_ada_scores

    # BorderlineSMOTE2
    X_borderline2, y_borderline2 = borderline2(X_train, y_train)

    borderline2_dt_scores = DecisionTree("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_dt_scores_total = borderline2_dt_scores_total + borderline2_dt_scores
    borderline2_rf_scores = RandomForest("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_rf_scores_total = borderline2_rf_scores_total + borderline2_rf_scores
    borderline2_svm_scores = SVM("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_svm_scores_total = borderline2_svm_scores_total + borderline2_svm_scores
    borderline2_nb_scores = NB("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_nb_scores_total = borderline2_nb_scores_total + borderline2_nb_scores
    borderline2_knn_scores = KNN("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_knn_scores_total = borderline2_knn_scores_total + borderline2_knn_scores
    borderline2_lr_scores = LR("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_lr_scores_total = borderline2_lr_scores_total + borderline2_lr_scores
    borderline2_ada_scores = Adaboost("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_ada_scores_total = borderline2_ada_scores_total + borderline2_ada_scores

    # ADASYN
    X_adasyn, y_adasyn = adasyn(X_train, y_train)

    adasyn_dt_scores = DecisionTree("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_dt_scores_total = adasyn_dt_scores_total + adasyn_dt_scores
    adasyn_rf_scores = RandomForest("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_rf_scores_total = adasyn_rf_scores_total + adasyn_rf_scores
    adasyn_svm_scores = SVM("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_svm_scores_total = adasyn_svm_scores_total + adasyn_svm_scores
    adasyn_nb_scores = NB("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_nb_scores_total = adasyn_nb_scores_total + adasyn_nb_scores
    adasyn_knn_scores = KNN("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_knn_scores_total = adasyn_knn_scores_total + adasyn_knn_scores
    adasyn_lr_scores = LR("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_lr_scores_total = adasyn_lr_scores_total + adasyn_lr_scores
    adasyn_ada_scores = Adaboost("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_ada_scores_total = adasyn_ada_scores_total + adasyn_ada_scores

    # TomekLinks
    X_tomeklinks, y_tomeklinks = tomeklinks(X_train, y_train)

    tomeklinks_dt_scores = DecisionTree("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_dt_scores_total = tomeklinks_dt_scores_total + tomeklinks_dt_scores
    tomeklinks_rf_scores = RandomForest("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_rf_scores_total = tomeklinks_rf_scores_total + tomeklinks_rf_scores
    tomeklinks_svm_scores = SVM("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_svm_scores_total = tomeklinks_svm_scores_total + tomeklinks_svm_scores
    tomeklinks_nb_scores = NB("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_nb_scores_total = tomeklinks_nb_scores_total + tomeklinks_nb_scores
    tomeklinks_knn_scores = KNN("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_knn_scores_total = tomeklinks_knn_scores_total + tomeklinks_knn_scores
    tomeklinks_lr_scores = LR("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_lr_scores_total = tomeklinks_lr_scores_total + tomeklinks_lr_scores
    tomeklinks_ada_scores = Adaboost("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
    tomeklinks_ada_scores_total = tomeklinks_ada_scores_total + tomeklinks_ada_scores

    # SMOTETomek
    X_smotetomek, y_smotetomek = smotetomek(X_train, y_train)

    smotetomek_dt_scores = DecisionTree("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_dt_scores_total = smotetomek_dt_scores_total + smotetomek_dt_scores
    smotetomek_rf_scores = RandomForest("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_rf_scores_total = smotetomek_rf_scores_total + smotetomek_rf_scores
    smotetomek_svm_scores = SVM("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_svm_scores_total = smotetomek_svm_scores_total + smotetomek_svm_scores
    smotetomek_nb_scores = NB("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_nb_scores_total = smotetomek_nb_scores_total + smotetomek_nb_scores
    smotetomek_knn_scores = KNN("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_knn_scores_total = smotetomek_knn_scores_total + smotetomek_knn_scores
    smotetomek_lr_scores = LR("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_lr_scores_total = smotetomek_lr_scores_total + smotetomek_lr_scores
    smotetomek_ada_scores = Adaboost("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_ada_scores_total = smotetomek_ada_scores_total + smotetomek_ada_scores

    # ImproveBorderline
    train = pd.concat([X_train, y_train], axis=1)

    synthetic = ImproveBorderline(train)
    # print(synthetic.shape)

    X_improve = np.r_[X_train, synthetic]
    y_improve = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic.shape[0]))]

    improve_dt_scores = DecisionTree("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_dt_scores_total = improve_dt_scores_total + improve_dt_scores
    improve_rf_scores = RandomForest("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_rf_scores_total = improve_rf_scores_total + improve_rf_scores
    improve_svm_scores = SVM("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_svm_scores_total = improve_svm_scores_total + improve_svm_scores
    improve_nb_scores = NB("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_nb_scores_total = improve_nb_scores_total + improve_nb_scores
    improve_knn_scores = KNN("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_knn_scores_total = improve_knn_scores_total + improve_knn_scores
    improve_lr_scores = LR("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_lr_scores_total = improve_lr_scores_total + improve_lr_scores
    improve_ada_scores = Adaboost("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    improve_ada_scores_total = improve_ada_scores_total + improve_ada_scores

    # ImproveBorderline2
    synthetic2 = ImproveBorderline2(train)
    # print(synthetic.shape)

    X_improve2 = np.r_[X_train, synthetic2]
    y_improve2 = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic2.shape[0]))]

    improve2_dt_scores = DecisionTree("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_dt_scores_total = improve2_dt_scores_total + improve2_dt_scores
    improve2_rf_scores = RandomForest("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_rf_scores_total = improve2_rf_scores_total + improve2_rf_scores
    improve2_svm_scores = SVM("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_svm_scores_total = improve2_svm_scores_total + improve2_svm_scores
    improve2_nb_scores = NB("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_nb_scores_total = improve2_nb_scores_total + improve2_nb_scores
    improve2_knn_scores = KNN("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_knn_scores_total = improve2_knn_scores_total + improve2_knn_scores
    improve2_lr_scores = LR("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_lr_scores_total = improve2_lr_scores_total + improve2_lr_scores
    improve2_ada_scores = Adaboost("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    improve2_ada_scores_total = improve2_ada_scores_total + improve2_ada_scores

# 输出
print("[F1 G_mean Auc]")
print('origin_dt_scores_mean', origin_dt_scores_total / fold)
print('origin_rf_scores_mean', origin_rf_scores_total / fold)
print('origin_svm_scores_mean', origin_svm_scores_total / fold)
print('origin_nb_scores_mean', origin_nb_scores_total / fold)
print('origin_knn_scores_mean', origin_knn_scores_total / fold)
print('origin_lr_scores_mean', origin_lr_scores_total / fold)
print('origin_ada_scores_mean', origin_ada_scores_total / fold)

print("=" * 20)
print('smote_dt_scores_mean', smote_dt_scores_total / fold)
print('smote_rf_scores_mean', smote_rf_scores_total / fold)
print('smote_svm_scores_mean', smote_svm_scores_total / fold)
print('smote_nb_scores_mean', smote_nb_scores_total / fold)
print('smote_knn_scores_mean', smote_knn_scores_total / fold)
print('smote_lr_scores_mean', smote_lr_scores_total / fold)
print('smote_ada_scores_mean', smote_ada_scores_total / fold)

print("=" * 20)
print('borderline_dt_scores_mean', borderline_dt_scores_total / fold)
print('borderline_rf_scores_mean', borderline_rf_scores_total / fold)
print('borderline_svm_scores_mean', borderline_svm_scores_total / fold)
print('borderline_nb_scores_mean', borderline_nb_scores_total / fold)
print('borderline_knn_scores_mean', borderline_knn_scores_total / fold)
print('borderline_lr_scores_mean', borderline_lr_scores_total / fold)
print('borderline_ada_scores_mean', borderline_ada_scores_total / fold)

print("=" * 20)
print('borderline2_dt_scores_mean', borderline2_dt_scores_total / fold)
print('borderline2_rf_scores_mean', borderline2_rf_scores_total / fold)
print('borderline2_svm_scores_mean', borderline2_svm_scores_total / fold)
print('borderline2_nb_scores_mean', borderline2_nb_scores_total / fold)
print('borderline2_knn_scores_mean', borderline2_knn_scores_total / fold)
print('borderline2_lr_scores_mean', borderline2_lr_scores_total / fold)
print('borderline2_ada_scores_mean', borderline2_ada_scores_total / fold)

print("=" * 20)
print('adasyn_dt_scores_mean', adasyn_dt_scores_total / fold)
print('adasyn_rf_scores_mean', adasyn_rf_scores_total / fold)
print('adasyn_svm_scores_mean', adasyn_svm_scores_total / fold)
print('adasyn_nb_scores_mean', adasyn_nb_scores_total / fold)
print('adasyn_knn_scores_mean', adasyn_knn_scores_total / fold)
print('adasyn_lr_scores_mean', adasyn_lr_scores_total / fold)
print('adasyn_ada_scores_mean', adasyn_ada_scores_total / fold)

print("=" * 20)
print('tomeklinks_dt_scores_mean', tomeklinks_dt_scores_total / fold)
print('tomeklinks_rf_scores_mean', tomeklinks_rf_scores_total / fold)
print('tomeklinks_svm_scores_mean', tomeklinks_svm_scores_total / fold)
print('tomeklinks_nb_scores_mean', tomeklinks_nb_scores_total / fold)
print('tomeklinks_knn_scores_mean', tomeklinks_knn_scores_total / fold)
print('tomeklinks_lr_scores_mean', tomeklinks_lr_scores_total / fold)
print('tomeklinks_ada_scores_mean', tomeklinks_ada_scores_total / fold)

print("=" * 20)
print('smotetomek_dt_scores_mean', smotetomek_dt_scores_total / fold)
print('smotetomek_rf_scores_mean', smotetomek_rf_scores_total / fold)
print('smotetomek_svm_scores_mean', smotetomek_svm_scores_total / fold)
print('smotetomek_nb_scores_mean', smotetomek_nb_scores_total / fold)
print('smotetomek_knn_scores_mean', smotetomek_knn_scores_total / fold)
print('smotetomek_lr_scores_mean', smotetomek_lr_scores_total / fold)
print('smotetomek_ada_scores_mean', smotetomek_ada_scores_total / fold)

print("=" * 20)
print('improve_dt_scores_mean', improve_dt_scores_total / fold)
print('improve_rf_scores_mean', improve_rf_scores_total / fold)
print('improve_svm_scores_mean', improve_svm_scores_total / fold)
print('improve_nb_scores_mean', improve_nb_scores_total / fold)
print('improve_knn_scores_mean', improve_knn_scores_total / fold)
print('improve_lr_scores_mean', improve_lr_scores_total / fold)
print('improve_ada_scores_mean', improve_ada_scores_total / fold)

print("=" * 20)
print('improve2_dt_scores_mean', improve2_dt_scores_total / fold)
print('improve2_rf_scores_mean', improve2_rf_scores_total / fold)
print('improve2_svm_scores_mean', improve2_svm_scores_total / fold)
print('improve2_nb_scores_mean', improve2_nb_scores_total / fold)
print('improve2_knn_scores_mean', improve2_knn_scores_total / fold)
print('improve2_lr_scores_mean', improve2_lr_scores_total / fold)
print('improve2_ada_scores_mean', improve2_ada_scores_total / fold)



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
#
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
# # TomekLinks
# X_tomeklinks, y_tomeklinks = tomeklinks(X_train, y_train)
#
# tomeklinks_dt_scores = DecisionTree("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# tomeklinks_rf_scores = RandomForest("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# tomeklinks_svm_scores = SVM("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# tomeklinks_nb_scores = NB("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# tomeklinks_knn_scores = KNN("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# tomeklinks_lr_scores = LR("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# tomeklinks_ada_scores = Adaboost("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_light_scores = Light("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
# # tomeklinks_xgb_scores = XGB("TomekLinks", X_tomeklinks, X_test, y_tomeklinks, y_test)
#
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
# # ImproveBorderline
# train = pd.concat([X_train, y_train], axis=1)
#
# synthetic = ImproveBorderline(train)
# # print(synthetic.shape)
#
# X_improve = np.r_[X_train, synthetic]
# y_improve = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic.shape[0]))]
#
# improve_dt_scores = DecisionTree("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_rf_scores = RandomForest("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_svm_scores = SVM("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_nb_scores = NB("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_knn_scores = KNN("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_lr_scores = LR("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# improve_ada_scores = Adaboost("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# # improve_light_scores = Light("ImproveBorderline", X_improve, X_test, y_improve, y_test)
# # improve_xgb_scores = XGB("ImproveBorderline", X_improve, X_test, y_improve, y_test)
#
# # ImproveBorderline2
# synthetic2 = ImproveBorderline2(train)
# # print(synthetic.shape)
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
# # improve2_light_scores = Light("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
# # improve2_xgb_scores = XGB("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)



