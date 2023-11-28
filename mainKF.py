from Read import *
from ImproveBorderline import *
from ImproveBorderline2 import *
from Classifier import *
from Sampling import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_moons
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

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

smotetomek_dt_scores_total = np.zeros(score_len)
smotetomek_rf_scores_total = np.zeros(score_len)
smotetomek_svm_scores_total = np.zeros(score_len)
smotetomek_nb_scores_total = np.zeros(score_len)
smotetomek_knn_scores_total = np.zeros(score_len)
smotetomek_lr_scores_total = np.zeros(score_len)
smotetomek_ada_scores_total = np.zeros(score_len)

smote_ipf_dt_scores_total = np.zeros(score_len)
smote_ipf_rf_scores_total = np.zeros(score_len)
smote_ipf_svm_scores_total = np.zeros(score_len)
smote_ipf_nb_scores_total = np.zeros(score_len)
smote_ipf_knn_scores_total = np.zeros(score_len)
smote_ipf_lr_scores_total = np.zeros(score_len)
smote_ipf_ada_scores_total = np.zeros(score_len)

rsmote_dt_scores_total = np.zeros(score_len)
rsmote_rf_scores_total = np.zeros(score_len)
rsmote_svm_scores_total = np.zeros(score_len)
rsmote_nb_scores_total = np.zeros(score_len)
rsmote_knn_scores_total = np.zeros(score_len)
rsmote_lr_scores_total = np.zeros(score_len)
rsmote_ada_scores_total = np.zeros(score_len)

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

data = shuttle_2_vs_5
# X, y = make_circles(n_samples=(650, 200), factor=0.2, noise=0.2, random_state=42)
# X, y = make_moons(n_samples=(650, 200), noise=0.2, random_state=42)

X = data.iloc[:, data.columns != "Class"]
y = data.iloc[:, data.columns == "Class"]

state_num = [40, 41, 42, 43, 44]

for state in state_num:
    print(f"random_state = {state}")

    # 区分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)

    # Origin
    # origin_dt_scores = DecisionTree("Origin", X_train, X_test, y_train, y_test)
    # origin_dt_scores_total = origin_dt_scores_total + origin_dt_scores
    # origin_rf_scores = RandomForest("Origin", X_train, X_test, y_train, y_test)
    # origin_rf_scores_total = origin_rf_scores_total + origin_rf_scores
    origin_svm_scores = SVM("Origin", X_train, X_test, y_train, y_test)
    origin_svm_scores_total = origin_svm_scores_total + origin_svm_scores
    # origin_nb_scores = NB("Origin", X_train, X_test, y_train, y_test)
    # origin_nb_scores_total = origin_nb_scores_total + origin_nb_scores
    origin_knn_scores = KNN("Origin", X_train, X_test, y_train, y_test)
    origin_knn_scores_total = origin_knn_scores_total + origin_knn_scores
    # origin_lr_scores = LR("Origin", X_train, X_test, y_train, y_test)
    # origin_lr_scores_total = origin_lr_scores_total + origin_lr_scores
    # origin_ada_scores = Adaboost("Origin", X_train, X_test, y_train, y_test)
    # origin_ada_scores_total = origin_ada_scores_total + origin_ada_scores

    # SMOTE
    X_smote, y_smote = smote(X_train, y_train)

    # smote_dt_scores = DecisionTree("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_dt_scores_total = smote_dt_scores_total + smote_dt_scores
    # smote_rf_scores = RandomForest("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_rf_scores_total = smote_rf_scores_total + smote_rf_scores
    smote_svm_scores = SVM("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_svm_scores_total = smote_svm_scores_total + smote_svm_scores
    # smote_nb_scores = NB("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_nb_scores_total = smote_nb_scores_total + smote_nb_scores
    smote_knn_scores = KNN("SMOTE", X_smote, X_test, y_smote, y_test)
    smote_knn_scores_total = smote_knn_scores_total + smote_knn_scores
    # smote_lr_scores = LR("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_lr_scores_total = smote_lr_scores_total + smote_lr_scores
    # smote_ada_scores = Adaboost("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_ada_scores_total = smote_ada_scores_total + smote_ada_scores

    # BorderlineSMOTE
    X_borderline, y_borderline = borderline(X_train, y_train)

    # borderline_dt_scores = DecisionTree("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_dt_scores_total = borderline_dt_scores_total + borderline_dt_scores
    # borderline_rf_scores = RandomForest("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_rf_scores_total = borderline_rf_scores_total + borderline_rf_scores
    borderline_svm_scores = SVM("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_svm_scores_total = borderline_svm_scores_total + borderline_svm_scores
    # borderline_nb_scores = NB("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_nb_scores_total = borderline_nb_scores_total + borderline_nb_scores
    borderline_knn_scores = KNN("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    borderline_knn_scores_total = borderline_knn_scores_total + borderline_knn_scores
    # borderline_lr_scores = LR("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_lr_scores_total = borderline_lr_scores_total + borderline_lr_scores
    # borderline_ada_scores = Adaboost("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_ada_scores_total = borderline_ada_scores_total + borderline_ada_scores

    # BorderlineSMOTE2
    X_borderline2, y_borderline2 = borderline2(X_train, y_train)

    # borderline2_dt_scores = DecisionTree("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_dt_scores_total = borderline2_dt_scores_total + borderline2_dt_scores
    # borderline2_rf_scores = RandomForest("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_rf_scores_total = borderline2_rf_scores_total + borderline2_rf_scores
    borderline2_svm_scores = SVM("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_svm_scores_total = borderline2_svm_scores_total + borderline2_svm_scores
    # borderline2_nb_scores = NB("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_nb_scores_total = borderline2_nb_scores_total + borderline2_nb_scores
    borderline2_knn_scores = KNN("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    borderline2_knn_scores_total = borderline2_knn_scores_total + borderline2_knn_scores
    # borderline2_lr_scores = LR("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_lr_scores_total = borderline2_lr_scores_total + borderline2_lr_scores
    # borderline2_ada_scores = Adaboost("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_ada_scores_total = borderline2_ada_scores_total + borderline2_ada_scores

    # ADASYN
    X_adasyn, y_adasyn = adasyn(X_train, y_train)

    # adasyn_dt_scores = DecisionTree("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_dt_scores_total = adasyn_dt_scores_total + adasyn_dt_scores
    # adasyn_rf_scores = RandomForest("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_rf_scores_total = adasyn_rf_scores_total + adasyn_rf_scores
    adasyn_svm_scores = SVM("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_svm_scores_total = adasyn_svm_scores_total + adasyn_svm_scores
    # adasyn_nb_scores = NB("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_nb_scores_total = adasyn_nb_scores_total + adasyn_nb_scores
    adasyn_knn_scores = KNN("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    adasyn_knn_scores_total = adasyn_knn_scores_total + adasyn_knn_scores
    # adasyn_lr_scores = LR("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_lr_scores_total = adasyn_lr_scores_total + adasyn_lr_scores
    # adasyn_ada_scores = Adaboost("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_ada_scores_total = adasyn_ada_scores_total + adasyn_ada_scores

    # SMOTETomek
    X_smotetomek, y_smotetomek = smotetomek(X_train, y_train)

    # smotetomek_dt_scores = DecisionTree("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_dt_scores_total = smotetomek_dt_scores_total + smotetomek_dt_scores
    # smotetomek_rf_scores = RandomForest("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_rf_scores_total = smotetomek_rf_scores_total + smotetomek_rf_scores
    smotetomek_svm_scores = SVM("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_svm_scores_total = smotetomek_svm_scores_total + smotetomek_svm_scores
    # smotetomek_nb_scores = NB("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_nb_scores_total = smotetomek_nb_scores_total + smotetomek_nb_scores
    smotetomek_knn_scores = KNN("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    smotetomek_knn_scores_total = smotetomek_knn_scores_total + smotetomek_knn_scores
    # smotetomek_lr_scores = LR("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_lr_scores_total = smotetomek_lr_scores_total + smotetomek_lr_scores
    # smotetomek_ada_scores = Adaboost("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_ada_scores_total = smotetomek_ada_scores_total + smotetomek_ada_scores

    # SMOTE_IPF
    X_train_ipf = np.array(X_train)
    y_train_ipf = np.array(y_train).flatten()
    X_smote_ipf, y_smote_ipf = smote_ipf(X_train_ipf, y_train_ipf)

    # smote_ipf_dt_scores = DecisionTree("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    # smote_ipf_dt_scores_total = smote_ipf_dt_scores_total + smote_ipf_dt_scores
    # smote_ipf_rf_scores = RandomForest("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    # smote_ipf_rf_scores_total = smote_ipf_rf_scores_total + smote_ipf_rf_scores
    smote_ipf_svm_scores = SVM("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    smote_ipf_svm_scores_total = smote_ipf_svm_scores_total + smote_ipf_svm_scores
    # smote_ipf_nb_scores = NB("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    # smote_ipf_nb_scores_total = smote_ipf_nb_scores_total + smote_ipf_nb_scores
    smote_ipf_knn_scores = KNN("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    smote_ipf_knn_scores_total = smote_ipf_knn_scores_total + smote_ipf_knn_scores
    # smote_ipf_lr_scores = LR("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    # smote_ipf_lr_scores_total = smote_ipf_lr_scores_total + smote_ipf_lr_scores
    # smote_ipf_ada_scores = Adaboost("SMOTE_IPF", X_smote_ipf, X_test, y_smote_ipf, y_test)
    # smote_ipf_ada_scores_total = smote_ipf_ada_scores_total + smote_ipf_ada_scores

    # RSMOTE_V8
    X_train_rsmote = np.array(X_train)
    y_train_rsmote = np.array(y_train).flatten()
    X_rsmote, y_rsmote = rsmote(X_train_rsmote, y_train_rsmote)

    # rsmote_dt_scores = DecisionTree("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    # rsmote_dt_scores_total = rsmote_dt_scores_total + rsmote_dt_scores
    # rsmote_rf_scores = RandomForest("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    # rsmote_rf_scores_total = rsmote_rf_scores_total + rsmote_rf_scores
    rsmote_svm_scores = SVM("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    rsmote_svm_scores_total = rsmote_svm_scores_total + rsmote_svm_scores
    # rsmote_nb_scores = NB("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    # rsmote_nb_scores_total = rsmote_nb_scores_total + rsmote_nb_scores
    rsmote_knn_scores = KNN("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    rsmote_knn_scores_total = rsmote_knn_scores_total + rsmote_knn_scores
    # rsmote_lr_scores = LR("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    # rsmote_lr_scores_total = rsmote_lr_scores_total + rsmote_lr_scores
    # rsmote_ada_scores = Adaboost("RSMOTE", X_rsmote, X_test, y_rsmote, y_test)
    # rsmote_ada_scores_total = rsmote_ada_scores_total + rsmote_ada_scores

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
    # improve_dt_scores_total = improve_dt_scores_total + improve_dt_scores
    # improve_rf_scores = RandomForest("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    # improve_rf_scores_total = improve_rf_scores_total + improve_rf_scores
    # improve_svm_scores = SVM("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    # improve_svm_scores_total = improve_svm_scores_total + improve_svm_scores
    # improve_nb_scores = NB("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    # improve_nb_scores_total = improve_nb_scores_total + improve_nb_scores
    # improve_knn_scores = KNN("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    # improve_knn_scores_total = improve_knn_scores_total + improve_knn_scores
    # improve_lr_scores = LR("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    # improve_lr_scores_total = improve_lr_scores_total + improve_lr_scores
    # improve_ada_scores = Adaboost("ImproveBorderline", X_improve, X_test, y_improve, y_test)
    # improve_ada_scores_total = improve_ada_scores_total + improve_ada_scores
    #
    # # ImproveBorderline2
    # synthetic2 = ImproveBorderline2(train)
    # # print(synthetic.shape)
    #
    # X_improve2 = np.r_[X_train, synthetic2]
    # y_improve2 = np.r_[np.array(y_train.values).flatten(), np.ones((synthetic2.shape[0]))]
    #
    # improve2_dt_scores = DecisionTree("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_dt_scores_total = improve2_dt_scores_total + improve2_dt_scores
    # improve2_rf_scores = RandomForest("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_rf_scores_total = improve2_rf_scores_total + improve2_rf_scores
    # improve2_svm_scores = SVM("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_svm_scores_total = improve2_svm_scores_total + improve2_svm_scores
    # improve2_nb_scores = NB("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_nb_scores_total = improve2_nb_scores_total + improve2_nb_scores
    # improve2_knn_scores = KNN("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_knn_scores_total = improve2_knn_scores_total + improve2_knn_scores
    # improve2_lr_scores = LR("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_lr_scores_total = improve2_lr_scores_total + improve2_lr_scores
    # improve2_ada_scores = Adaboost("ImproveBorderline2", X_improve2, X_test, y_improve2, y_test)
    # improve2_ada_scores_total = improve2_ada_scores_total + improve2_ada_scores


# 输出
print("[F1 G_mean Auc]")
# print('origin_dt_scores_mean', origin_dt_scores_total / len(state_num))
# print('origin_rf_scores_mean', origin_rf_scores_total / len(state_num))
print('origin_svm_scores_mean', origin_svm_scores_total / len(state_num))
# print('origin_nb_scores_mean', origin_nb_scores_total / len(state_num))
print('origin_knn_scores_mean', origin_knn_scores_total / len(state_num))
# print('origin_lr_scores_mean', origin_lr_scores_total / len(state_num))
# print('origin_ada_scores_mean', origin_ada_scores_total / len(state_num))

print("=" * 20)
# print('smote_dt_scores_mean', smote_dt_scores_total / len(state_num))
# print('smote_rf_scores_mean', smote_rf_scores_total / len(state_num))
print('smote_svm_scores_mean', smote_svm_scores_total / len(state_num))
# print('smote_nb_scores_mean', smote_nb_scores_total / len(state_num))
print('smote_knn_scores_mean', smote_knn_scores_total / len(state_num))
# print('smote_lr_scores_mean', smote_lr_scores_total / len(state_num))
# print('smote_ada_scores_mean', smote_ada_scores_total / len(state_num))

print("=" * 20)
# print('borderline_dt_scores_mean', borderline_dt_scores_total / len(state_num))
# print('borderline_rf_scores_mean', borderline_rf_scores_total / len(state_num))
print('borderline_svm_scores_mean', borderline_svm_scores_total / len(state_num))
# print('borderline_nb_scores_mean', borderline_nb_scores_total / len(state_num))
print('borderline_knn_scores_mean', borderline_knn_scores_total / len(state_num))
# print('borderline_lr_scores_mean', borderline_lr_scores_total / len(state_num))
# print('borderline_ada_scores_mean', borderline_ada_scores_total / len(state_num))

print("=" * 20)
# print('borderline2_dt_scores_mean', borderline2_dt_scores_total / len(state_num))
# print('borderline2_rf_scores_mean', borderline2_rf_scores_total / len(state_num))
print('borderline2_svm_scores_mean', borderline2_svm_scores_total / len(state_num))
# print('borderline2_nb_scores_mean', borderline2_nb_scores_total / len(state_num))
print('borderline2_knn_scores_mean', borderline2_knn_scores_total / len(state_num))
# print('borderline2_lr_scores_mean', borderline2_lr_scores_total / len(state_num))
# print('borderline2_ada_scores_mean', borderline2_ada_scores_total / len(state_num))

print("=" * 20)
# print('adasyn_dt_scores_mean', adasyn_dt_scores_total / len(state_num))
# print('adasyn_rf_scores_mean', adasyn_rf_scores_total / len(state_num))
print('adasyn_svm_scores_mean', adasyn_svm_scores_total / len(state_num))
# print('adasyn_nb_scores_mean', adasyn_nb_scores_total / len(state_num))
print('adasyn_knn_scores_mean', adasyn_knn_scores_total / len(state_num))
# print('adasyn_lr_scores_mean', adasyn_lr_scores_total / len(state_num))
# print('adasyn_ada_scores_mean', adasyn_ada_scores_total / len(state_num))

print("=" * 20)
# print('smotetomek_dt_scores_mean', smotetomek_dt_scores_total / len(state_num))
# print('smotetomek_rf_scores_mean', smotetomek_rf_scores_total / len(state_num))
print('smotetomek_svm_scores_mean', smotetomek_svm_scores_total / len(state_num))
# print('smotetomek_nb_scores_mean', smotetomek_nb_scores_total / len(state_num))
print('smotetomek_knn_scores_mean', smotetomek_knn_scores_total / len(state_num))
# print('smotetomek_lr_scores_mean', smotetomek_lr_scores_total / len(state_num))
# print('smotetomek_ada_scores_mean', smotetomek_ada_scores_total / len(state_num))

print("=" * 20)
# print('smote_ipf_dt_scores_mean', smote_ipf_dt_scores_total / len(state_num))
# print('smote_ipf_rf_scores_mean', smote_ipf_rf_scores_total / len(state_num))
print('smote_ipf_svm_scores_mean', smote_ipf_svm_scores_total / len(state_num))
# print('smote_ipf_nb_scores_mean', smote_ipf_nb_scores_total / len(state_num))
print('smote_ipf_knn_scores_mean', smote_ipf_knn_scores_total / len(state_num))
# print('smote_ipf_lr_scores_mean', smote_ipf_lr_scores_total / len(state_num))
# print('smote_ipf_ada_scores_mean', smote_ipf_ada_scores_total / len(state_num))

print("=" * 20)
# print('rsmote_dt_scores_mean', rsmote_dt_scores_total / len(state_num))
# print('rsmote_rf_scores_mean', rsmote_rf_scores_total / len(state_num))
print('rsmote_svm_scores_mean', rsmote_svm_scores_total / len(state_num))
# print('rsmote_nb_scores_mean', rsmote_nb_scores_total / len(state_num))
print('rsmote_knn_scores_mean', rsmote_knn_scores_total / len(state_num))
# print('rsmote_lr_scores_mean', rsmote_lr_scores_total / len(state_num))
# print('rsmote_ada_scores_mean', rsmote_ada_scores_total / len(state_num))

# print("=" * 20)
# print('improve_dt_scores_mean', improve_dt_scores_total / len(state_num))
# print('improve_rf_scores_mean', improve_rf_scores_total / len(state_num))
# print('improve_svm_scores_mean', improve_svm_scores_total / len(state_num))
# print('improve_nb_scores_mean', improve_nb_scores_total / len(state_num))
# print('improve_knn_scores_mean', improve_knn_scores_total / len(state_num))
# print('improve_lr_scores_mean', improve_lr_scores_total / len(state_num))
# print('improve_ada_scores_mean', improve_ada_scores_total / len(state_num))
#
# print("=" * 20)
# print('improve2_dt_scores_mean', improve2_dt_scores_total / len(state_num))
# print('improve2_rf_scores_mean', improve2_rf_scores_total / len(state_num))
# print('improve2_svm_scores_mean', improve2_svm_scores_total / len(state_num))
# print('improve2_nb_scores_mean', improve2_nb_scores_total / len(state_num))
# print('improve2_knn_scores_mean', improve2_knn_scores_total / len(state_num))
# print('improve2_lr_scores_mean', improve2_lr_scores_total / len(state_num))
# print('improve2_ada_scores_mean', improve2_ada_scores_total / len(state_num))
