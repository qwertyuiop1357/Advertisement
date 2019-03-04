from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn.metrics import roc_auc_score
import xlearn as xl
from sklearn.datasets import dump_svmlight_file

if __name__ == '__main__':
    # logistic regression 
    c_param_range = 1
    log = LogisticRegression(C = c_param_range, class_weight='balanced').fit(X_train, y_train)
    pre = log.predict(X_test)
    auc = roc_auc_score(y_test, pre)
    print("ROC auc score for logistic regression is " + auc)
    
    
    # XGBoost
    xgb = xgboost.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=50, silent=True, 
                          objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, 
                          gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
                          colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
                          base_score=0.5, random_state=0, seed=None, missing=None)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['auc'],
                  early_stopping_rounds=20,verbose=False)

    evals_result = xgb.evals_result()
    l = evals_result['validation_0']['auc']
    auc = sum(l) / float(len(l))
    print("ROC auc score for XGBoost is " + auc)
    
    # XGBoost+LR
    res = xgb.apply(X_train)
    test_res = xgb.apply(X_test)

    log = LogisticRegression(C = c_param_range, class_weight='balanced').fit(res, y_train)
    pre = log.predict(test_res)
    auc = roc_auc_score(y_test, pre)
    print("ROC auc score for XGBoost+LR is " + auc)
    
    # XGBoost+FM
    res = pd.DataFrame(res)
    test_res = pd.DataFrame(test_res)
    dummy = pd.get_dummies(res)
    mat = dummy.as_matrix()
    dump_svmlight_file(mat, y_train, 'svm-output.libsvm')

    dummy_1 = pd.get_dummies(test_res)
    mat_1 = dummy_1.as_matrix()
    dump_svmlight_file(mat_1, y_test, 'svm-output_1.libsvm')
    fm_model = xl.create_fm() 
    fm_model.setTrain('svm-output.libsvm')  

    param = {'task':'binary', 'lr':0.1, 'lambda':0.001, 'metric':'auc'}

    fm_model.fit(param, './model.out')
    fm_model.setTest('svm-output_1.libsvm')  
    fm_model.setSigmoid()

    fm_model.predict("./model.out", "./output.txt")
    dfm = pd.read_fwf('output.txt', header=None)
    auc = roc_auc_score(y_test, dfm)
    print("ROC auc score for XGBoost+FM is " + auc)
    
    # XGBoost+FFM
    ffm_train = FFMFormatPandas()
    ffm_train_data = ffm_train.fit_transform(df, y='clicked')
    ffm_model = xl.create_ffm() 
    ffm_model.setTrain('svm-output.libffm')  

    param = {'task':'binary', 'lr':0.1, 'lambda':0.001, 'metric':'auc'}

    ffm_model.fit(param, './model.out')
    ffm_model.setTest('svm-output_1.libffm')  
    ffm_model.setSigmoid()

    fm_model.predict("./model.out", "./output.txt")
    dfm = pd.read_fwf('output.txt', header=None)
    auc = roc_auc_score(y_test, dfm)
    print("ROC auc score for XGBoost+FFM is " + auc)
