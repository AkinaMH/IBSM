from Read import *
from ImproveBorderline import *
from ImproveBorderline2 import *
from Classifier import *
from Sampling import *
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


score_len = 2
origin_dt_scores_total = np.zeros(score_len)
origin_rf_scores_total = np.zeros(score_len)
origin_svm_scores_total = np.zeros(score_len)

smote_dt_scores_total = np.zeros(score_len)
smote_rf_scores_total = np.zeros(score_len)
smote_svm_scores_total = np.zeros(score_len)

borderline_dt_scores_total = np.zeros(score_len)
borderline_rf_scores_total = np.zeros(score_len)
borderline_svm_scores_total = np.zeros(score_len)

borderline2_dt_scores_total = np.zeros(score_len)
borderline2_rf_scores_total = np.zeros(score_len)
borderline2_svm_scores_total = np.zeros(score_len)

adasyn_dt_scores_total = np.zeros(score_len)
adasyn_rf_scores_total = np.zeros(score_len)
adasyn_svm_scores_total = np.zeros(score_len)

kmsmote_dt_scores_total = np.zeros(score_len)
kmsmote_rf_scores_total = np.zeros(score_len)
kmsmote_svm_scores_total = np.zeros(score_len)

oss_dt_scores_total = np.zeros(score_len)
oss_rf_scores_total = np.zeros(score_len)
oss_svm_scores_total = np.zeros(score_len)

smotetomek_dt_scores_total = np.zeros(score_len)
smotetomek_rf_scores_total = np.zeros(score_len)
smotetomek_svm_scores_total = np.zeros(score_len)

improve_dt_scores_total = np.zeros(score_len)
improve_rf_scores_total = np.zeros(score_len)
improve_svm_scores_total = np.zeros(score_len)

improve2_dt_scores_total = np.zeros(score_len)
improve2_rf_scores_total = np.zeros(score_len)
improve2_svm_scores_total = np.zeros(score_len)


result = []
# file = pima_filenames
# file = yeast3_filenames
# file = iris0_filenames  报错
# file = ecoli4_filenames
file = wisconsin_filenames

for i in range(len(file))[::2]:
    if i+1 < len(file):
        result.append((file[i], file[i+1]))
    else:
        result.append((file[i], ))

    train = pd.read_csv(result[0][0])
    test = pd.read_csv(result[0][1])

    X_train = train.iloc[:, train.columns != "Class"]
    y_train = train.iloc[:, train.columns == "Class"]
    X_test = test.iloc[:, test.columns != "Class"]
    y_test = test.iloc[:, test.columns == "Class"]

    # origin_dt_scores = DecisionTree("Origin", X_train, X_test, y_train, y_test)
    # origin_dt_scores_total = origin_dt_scores_total + origin_dt_scores
    # origin_rf_scores = RandomForest("Origin", X_train, X_test, y_train, y_test)
    # origin_rf_scores_total = origin_rf_scores_total + origin_rf_scores
    # origin_svm_scores = SVM("Origin", X_train, X_test, y_train, y_test)
    # origin_svm_scores_total = origin_svm_scores_total + origin_svm_scores

    # SMOTE
    # X_smote, y_smote = smote(X_train, y_train)
    #
    # smote_dt_scores = DecisionTree("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_dt_scores_total = smote_dt_scores_total + smote_dt_scores
    # smote_rf_scores = RandomForest("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_rf_scores_total = smote_rf_scores_total + smote_rf_scores
    # smote_svm_scores = SVM("SMOTE", X_smote, X_test, y_smote, y_test)
    # smote_svm_scores_total = smote_svm_scores_total + smote_svm_scores

    # BorderlineSMOTE
    # X_borderline, y_borderline = borderline(X_train, y_train)
    #
    # borderline_dt_scores = DecisionTree("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_dt_scores_total = borderline_dt_scores_total + borderline_dt_scores
    # borderline_rf_scores = RandomForest("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_rf_scores_total = borderline_rf_scores_total + borderline_rf_scores
    # borderline_svm_scores = SVM("BorderlineSMOTE", X_borderline, X_test, y_borderline, y_test)
    # borderline_svm_scores_total = borderline_svm_scores_total + borderline_svm_scores

    # BorderlineSMOTE2
    # X_borderline2, y_borderline2 = borderline2(X_train, y_train)
    #
    # borderline2_dt_scores = DecisionTree("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_dt_scores_total = borderline2_dt_scores_total + borderline2_dt_scores
    # borderline2_rf_scores = RandomForest("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_rf_scores_total = borderline2_rf_scores_total + borderline2_rf_scores
    # borderline2_svm_scores = SVM("BorderlineSMOTE2", X_borderline2, X_test, y_borderline2, y_test)
    # borderline2_svm_scores_total = borderline2_svm_scores_total + borderline2_svm_scores

    # ADASYN
    # X_adasyn, y_adasyn = adasyn(X_train, y_train)
    #
    # adasyn_dt_scores = DecisionTree("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_dt_scores_total = adasyn_dt_scores_total + adasyn_dt_scores
    # adasyn_rf_scores = RandomForest("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_rf_scores_total = adasyn_rf_scores_total + adasyn_rf_scores
    # adasyn_svm_scores = SVM("ADASYN", X_adasyn, X_test, y_adasyn, y_test)
    # adasyn_svm_scores_total = adasyn_svm_scores_total + adasyn_svm_scores

    # KMeansSMOTE
    # X_kmsmote, y_kmsmote = kmsmote(X_train, y_train)
    #
    # kmsmote_dt_scores = DecisionTree("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
    # kmsmote_dt_scores_total = kmsmote_dt_scores_total + kmsmote_dt_scores
    # kmsmote_rf_scores = RandomForest("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
    # kmsmote_rf_scores_total = kmsmote_rf_scores_total + kmsmote_rf_scores
    # kmsmote_svm_scores = SVM("KMeansSMOTE", X_kmsmote, X_test, y_kmsmote, y_test)
    # kmsmote_svm_scores_total = kmsmote_svm_scores_total + kmsmote_svm_scores

    # OneSidedSelection
    # X_oss, y_oss = onesidedselection(X_train, y_train)
    #
    # oss_dt_scores = DecisionTree("OneSidedSelection", X_oss, X_test, y_oss, y_test)
    # oss_dt_scores_total = oss_dt_scores_total + oss_dt_scores
    # oss_rf_scores = RandomForest("OneSidedSelection", X_oss, X_test, y_oss, y_test)
    # oss_rf_scores_total = oss_rf_scores_total + oss_rf_scores
    # oss_svm_scores = SVM("OneSidedSelection", X_oss, X_test, y_oss, y_test)
    # oss_svm_scores_total = oss_svm_scores_total + oss_svm_scores

    # SMOTETomek
    # X_smotetomek, y_smotetomek = smotetomek(X_train, y_train)
    #
    # smotetomek_dt_scores = DecisionTree("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_dt_scores_total = smotetomek_dt_scores_total + smotetomek_dt_scores
    # smotetomek_rf_scores = RandomForest("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_rf_scores_total = smotetomek_rf_scores_total + smotetomek_rf_scores
    # smotetomek_svm_scores = SVM("SMOTETomek", X_smotetomek, X_test, y_smotetomek, y_test)
    # smotetomek_svm_scores_total = smotetomek_svm_scores_total + smotetomek_svm_scores

    # ImproveBorderline
    # IB_train = pd.concat([X_train, y_train], axis=1)

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


# 输出
print("[F1 G_mean]")
print('origin_dt_scores_mean', origin_dt_scores_total / 5)
print('origin_rf_scores_mean', origin_rf_scores_total / 5)
print('origin_svm_scores_mean', origin_svm_scores_total / 5)

print("=" * 20)
print('smote_dt_scores_mean', smote_dt_scores_total / 5)
print('smote_rf_scores_mean', smote_rf_scores_total / 5)
print('smote_svm_scores_mean', smote_svm_scores_total / 5)

print("=" * 20)
print('borderline_dt_scores_mean', borderline_dt_scores_total / 5)
print('borderline_rf_scores_mean', borderline_rf_scores_total / 5)
print('borderline_svm_scores_mean', borderline_svm_scores_total / 5)

print("=" * 20)
print('borderline2_dt_scores_mean', borderline2_dt_scores_total / 5)
print('borderline2_rf_scores_mean', borderline2_rf_scores_total / 5)
print('borderline2_svm_scores_mean', borderline2_svm_scores_total / 5)

print("=" * 20)
print('adasyn_dt_scores_mean', adasyn_dt_scores_total / 5)
print('adasyn_rf_scores_mean', adasyn_rf_scores_total / 5)
print('adasyn_svm_scores_mean', adasyn_svm_scores_total / 5)

print("=" * 20)
print('kmsmote_dt_scores_mean', kmsmote_dt_scores_total / 5)
print('kmsmote_rf_scores_mean', kmsmote_rf_scores_total / 5)
print('kmsmote_svm_scores_mean', kmsmote_svm_scores_total / 5)

print("=" * 20)
print('oss_dt_scores_mean', oss_dt_scores_total / 5)
print('oss_rf_scores_mean', oss_rf_scores_total / 5)
print('oss_svm_scores_mean', oss_svm_scores_total / 5)

print("=" * 20)
print('smotetomek_dt_scores_mean', smotetomek_dt_scores_total / 5)
print('smotetomek_rf_scores_mean', smotetomek_rf_scores_total / 5)
print('smotetomek_svm_scores_mean', smotetomek_svm_scores_total / 5)

print("=" * 20)
print('improve_dt_scores_mean', improve_dt_scores_total / 5)
print('improve_rf_scores_mean', improve_rf_scores_total / 5)
print('improve_svm_scores_mean', improve_svm_scores_total / 5)

print("=" * 20)
print('improve2_dt_scores_mean', improve2_dt_scores_total / 5)
print('improve2_rf_scores_mean', improve2_rf_scores_total / 5)
print('improve2_svm_scores_mean', improve2_svm_scores_total / 5)




