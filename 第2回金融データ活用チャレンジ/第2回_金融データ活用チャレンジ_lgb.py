#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, KFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
import optuna
import seaborn as sns
import datetime


# In[ ]:


def for_train_target_encode(df, column, target):
    tmp = np.full(df.shape[0], np.nan)
    enc_kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
    for tr_idx, val_idx in enc_kf.split(df):
        target_mean = df.iloc[tr_idx].groupby(column)[target].mean()
        tmp[val_idx] = df[column].iloc[val_idx].map(target_mean)
    df[column] = tmp
    return df[column]
def for_test_target_encode(train, test, column, target):
    target_mean = train.groupby(column)[target].mean()
    test[column] = test[column].map(target_mean)
    return test[column]

def FranchiseCode_classify(value):
    if value == 0:
        return 0
    elif value == 1:
        return 1
    elif 2 <= value < 1000:
        return 1000
    elif 1000 <= value < 9999:
        return 9000
    elif 10000 <= value < 20000:
        return 10000
    elif 20000 <= value < 30000:
        return 20000
    elif 30000 <= value < 40000:
        return 30000
    elif 40000 <= value < 50000:
        return 40000
    elif 50000 <= value < 60000:
        return 50000
    elif 60000 <= value < 70000:
        return 60000
    elif 70000 <= value < 80000:
        return 70000
    elif 80000 <= value < 90000:
        return 80000
    elif 90000 <= value <= 99999:
        return 90000
    else:
        return "Undefined"

def zero_one_classify(value):
    if value == 0:
        return 0
    else:
        return 1

def check_same_category(row):
    if row['State'] == row['BankState']:
        return 1
    else:
        return 0
    
def divide_columns(row, col1, col2):
    if row[col2] == 0:
        return row[col1]
    else:
        return row[col1] / row[col2]

train = pd.read_csv("Desktop/コンペ/第2回 金融データ活用チャレンジ/train.csv", index_col = 0)
test = pd.read_csv("Desktop/コンペ/第2回 金融データ活用チャレンジ/test.csv", index_col = 0)
submit = pd.read_csv("Desktop/コンペ/第2回 金融データ活用チャレンジ/sample_submission.csv", header = None, index_col=0,)
train[['DisbursementGross',  'GrAppv', 'SBA_Appv']] = train[['DisbursementGross',  'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', '').replace('.00', ''))
train[['DisbursementGross',  'GrAppv', 'SBA_Appv']] = train[['DisbursementGross',  'GrAppv', 'SBA_Appv']].astype(int)
test[['DisbursementGross',  'GrAppv', 'SBA_Appv']]=test[['DisbursementGross',  'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', '').replace('.00', ''))
test[['DisbursementGross',  'GrAppv', 'SBA_Appv']] = test[['DisbursementGross',  'GrAppv', 'SBA_Appv']].astype(int)

train['ApprovalDate'] = pd.to_datetime(train['ApprovalDate'], format='%d-%b-%y')
test['ApprovalDate'] = pd.to_datetime(test['ApprovalDate'], format='%d-%b-%y')
train['DisbursementDate'] = pd.to_datetime(train['DisbursementDate'], format='%d-%b-%y', errors='coerce')
test['DisbursementDate'] = pd.to_datetime(test['DisbursementDate'], format='%d-%b-%y', errors='coerce')
train["date_diff"] = (train['DisbursementDate'] - train['ApprovalDate']).dt.days
test["date_diff"] = (test['DisbursementDate'] - test['ApprovalDate']).dt.days

train['ApprovalDate'] = train['ApprovalDate'].dt.strftime('%Y/%m/%d')
test['ApprovalDate'] = test['ApprovalDate'].dt.strftime('%Y/%m/%d')
train['DisbursementDate'] = train['DisbursementDate'].dt.strftime('%Y/%m/%d')
test['DisbursementDate'] = test['DisbursementDate'].dt.strftime('%Y/%m/%d')


train["ApprovalM"] = train["ApprovalDate"].apply(lambda x: int(x.split("/")[1]))
test["ApprovalM"] = test["ApprovalDate"].apply(lambda x: int(x.split("/")[1]))
train["ApprovalD"] = train["ApprovalDate"].apply(lambda x: int(x.split("/")[2]))
test["ApprovalD"] = test["ApprovalDate"].apply(lambda x: int(x.split("/")[2]))

train["Sector"] = train["Sector"].astype(str)
test["Sector"] = test["Sector"].astype(str)

train["job_sum"] = train["CreateJob"] + train["RetainedJob"]
test["job_sum"] = test["CreateJob"] + test["RetainedJob"]

train["job_total"] = train["CreateJob"] + train["RetainedJob"] + train["NoEmp"]
test["job_total"] = test["CreateJob"] + test["RetainedJob"] + test["NoEmp"]

train["FranchiseCode_ec"] = train["FranchiseCode"].apply(FranchiseCode_classify)
test["FranchiseCode_ec"] = test["FranchiseCode"].apply(FranchiseCode_classify)

train["St/BSt"] = train.apply(check_same_category, axis = 1)
test["St/BSt"] = test.apply(check_same_category, axis = 1)

continuous_feature = ["Term", "NoEmp", "CreateJob", "RetainedJob", "DisbursementGross", "GrAppv", "SBA_Appv"]
for col_1 in continuous_feature:
    for col_2 in continuous_feature:
        for col_3 in continuous_feature:
            if col_1 == col_2:
                pass
            elif col_2 == col_3:
                pass
            elif col_3 == col_1:
                pass
            else:
                train[col_1 + "/" + col_2] = train.apply(divide_columns, args=(col_1, col_2), axis=1)
                test[col_1 + "/" + col_2] = test.apply(divide_columns, args=(col_1, col_2), axis=1)
                train[col_1 + "+" + col_2] = train[col_1] + train[col_2]
                test[col_1 + "+" + col_2] = test[col_1] + test[col_2]
                train[col_1 + "*" + col_2] = train[col_1] * train[col_2]
                test[col_1 + "*" + col_2] = test[col_1] * test[col_2]
                train[col_1 + "-" + col_2] = train[col_1] - train[col_2]
                test[col_1 + "-" + col_2] = test[col_1] - test[col_2]
                train[col_1 + "/" + col_2 + "/" + col_3] = train.apply(divide_columns, args=(col_1 + "/" + col_2, col_3), axis=1)
                train[col_1 + "/" + col_2 + "+" + col_3] = train[col_1 + "/" + col_2] + train[col_3]
                train[col_1 + "/" + col_2 + "-" + col_3] = train[col_1 + "/" + col_2] - train[col_3]
                train[col_1 + "/" + col_2 + "*" + col_3] = train[col_1 + "/" + col_2] * train[col_3]
                train[col_1 + "+" + col_2 + "/" + col_3] = train.apply(divide_columns, args=(col_1 + "+" + col_2, col_3), axis=1)
                train[col_1 + "+" + col_2 + "+" + col_3] = train[col_1] + train[col_2] + train[col_3]
                train[col_1 + "+" + col_2 + "-" + col_3] = train[col_1] + train[col_2] - train[col_3]
                train[col_1 + "+" + col_2 + "*" + col_3] = (train[col_1] + train[col_2]) * train[col_3]
                train[col_1 + "-" + col_2 + "/" + col_3] = train.apply(divide_columns, args=(col_1 + "-" + col_2, col_3), axis=1)
                train[col_1 + "-" + col_2 + "+" + col_3] = train[col_1] - train[col_2] + train[col_3]
                train[col_1 + "-" + col_2 + "-" + col_3] = train[col_1] - train[col_2] - train[col_3]
                train[col_1 + "-" + col_2 + "*" + col_3] = (train[col_1] - train[col_2]) * train[col_3]
                train[col_1 + "*" + col_2 + "/" + col_3] = train.apply(divide_columns, args=(col_1 + "*" + col_2, col_3), axis=1)
                train[col_1 + "*" + col_2 + "+" + col_3] = train[col_1] * train[col_2] + train[col_3]
                train[col_1 + "*" + col_2 + "-" + col_3] = train[col_1] * train[col_2] - train[col_3]
                train[col_1 + "*" + col_2 + "*" + col_3] = (train[col_1] * train[col_2]) * train[col_3]                
                test[col_1 + "/" + col_2 + "/" + col_3] = test.apply(divide_columns, args=(col_1 + "/" + col_2, col_3), axis=1)
                test[col_1 + "/" + col_2 + "+" + col_3] = test[col_1 + "/" + col_2] + test[col_3]
                test[col_1 + "/" + col_2 + "-" + col_3] = test[col_1 + "/" + col_2] - test[col_3]
                test[col_1 + "/" + col_2 + "*" + col_3] = test[col_1 + "/" + col_2] * test[col_3]
                test[col_1 + "+" + col_2 + "/" + col_3] = test.apply(divide_columns, args=(col_1 + "+" + col_2, col_3), axis=1)
                test[col_1 + "+" + col_2 + "+" + col_3] = test[col_1] + test[col_2] + test[col_3]
                test[col_1 + "+" + col_2 + "-" + col_3] = test[col_1] + test[col_2] - test[col_3]
                test[col_1 + "+" + col_2 + "*" + col_3] = (test[col_1] + test[col_2]) * test[col_3]
                test[col_1 + "-" + col_2 + "/" + col_3] = test.apply(divide_columns, args=(col_1 + "-" + col_2, col_3), axis=1)
                test[col_1 + "-" + col_2 + "+" + col_3] = test[col_1] - test[col_2] + test[col_3]
                test[col_1 + "-" + col_2 + "-" + col_3] = test[col_1] - test[col_2] - test[col_3]
                test[col_1 + "-" + col_2 + "*" + col_3] = (test[col_1] - test[col_2]) * test[col_3]
                test[col_1 + "*" + col_2 + "/" + col_3] = test.apply(divide_columns, args=(col_1 + "*" + col_2, col_3), axis=1)
                test[col_1 + "*" + col_2 + "+" + col_3] = test[col_1] * test[col_2] + test[col_3]
                test[col_1 + "*" + col_2 + "-" + col_3] = test[col_1] * test[col_2] - test[col_3]
                test[col_1 + "*" + col_2 + "*" + col_3] = (test[col_1] * test[col_2]) * test[col_3]    


for col_1 in['FranchiseCode_ec', "State", "BankState", "Sector", "NewExist", "UrbanRural", "RevLineCr", "LowDoc", 'ApprovalFY', 'ApprovalM', 'ApprovalD']:
    df = train[[col_1, "Term", "DisbursementGross", "GrAppv", "SBA_Appv", "job_total", "job_sum"]].copy()
    temp = df.groupby(col_1)[["Term", "DisbursementGross", "GrAppv", "SBA_Appv", "job_total", "job_sum"]].agg(["mean", "median", "std", "max"])
    for col_2 in ["Term", "DisbursementGross", "GrAppv", "SBA_Appv", "job_total", "job_sum"]:
        train[col_1+ "_" + col_2 + "_mean"] = train[col_1].map(temp[col_2, "mean"])
        test[col_1+ "_" + col_2 + "_mean"] = test[col_1].map(temp[col_2, "mean"])
        train[col_1+ "_" + col_2 + "_med"] = train[col_1].map(temp[col_2, "median"])
        test[col_1+ "_" + col_2 + "_med"] = test[col_1].map(temp[col_2, "median"])
        train[col_1+ "_" + col_2 + "_std"] = train[col_1].map(temp[col_2, "std"])
        test[col_1+ "_" + col_2 + "_std"] = test[col_1].map(temp[col_2, "std"])
        train[col_1+ "_" + col_2 + "_max"] = train[col_1].map(temp[col_2, "max"])
        test[col_1+ "_" + col_2 + "_max"] = test[col_1].map(temp[col_2, "max"])

categorical_columns = ['City', 'State', 'BankState', "RevLineCr", "LowDoc", "Sector", "FranchiseCode"]
for column in categorical_columns:
    test[column] = for_test_target_encode(train, test, column, "MIS_Status")
for column in categorical_columns:
    train[column] = for_train_target_encode(train, column, "MIS_Status")


# In[ ]:


def lgb_get_feature_importances(X, y, shuffle=False):
    if shuffle:
        y = np.random.permutation(y)
        
    lgb = LGBMClassifier(random_state = 0, class_weight = {0: 1.5, 1: 1.0})
    lgb.fit(X,y)

    imp_df = pd.DataFrame()
    imp_df["feature"] = X.columns
    imp_df["importance"] = lgb.feature_importances_
    return imp_df.sort_values("importance", ascending=False)


y_train = train["MIS_Status"]
X_train = train.drop(["DisbursementDate", "ApprovalDate", "MIS_Status", "FranchiseCode_ec"], axis = 1)

lgb_actual_imp_df = lgb_get_feature_importances(X_train, y_train, shuffle=False)

N = 100
lgb_null_imp_df = pd.DataFrame()
for i in range(N):
    lgb_imp_df = lgb_get_feature_importances(X_train, y_train, shuffle=True)
    lgb_imp_df["run"] = i + 1
    lgb_null_imp_df = pd.concat([lgb_null_imp_df, lgb_imp_df])


# In[ ]:


threshold = 78

lgb_imp_features = []
for feature in lgb_actual_imp_df["feature"]:
    actual_value = lgb_actual_imp_df.query(f"feature=='{feature}'")["importance"].values
    null_value = lgb_null_imp_df.query(f"feature=='{feature}'")["importance"].values
    percentage = (null_value < actual_value).sum() / null_value.size * 100
    if percentage >= threshold:
        lgb_imp_features.append(feature)

print(len(lgb_imp_features))
lgb_imp_features


# In[ ]:


y_train = train["MIS_Status"]
X_train = train[lgb_imp_features]
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
params={'num_iterations': 4765, 'learning_rate': 0.009799855546680735, 'max_depth': 8, 'reg_alpha': 34.927535315330154, 'reg_lambda': 116.82707823139178, 'min_child_samples': 572, 'num_leaves': 4137, 'max_bin': 101, 'class_weight':{0: 1.2610662662270764, 1: 1.0}}

scores = []
pred_Ser = pd.Series(np.full(X_train.shape[0], -1))
for tr_idx, val_idx in kf.split(X_train, y_train):
    X_train_train, X_train_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_train_train, y_train_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    pred = []
    for i in range(15):
        rus = RandomUnderSampler(random_state=i, replacement=True, sampling_strategy = {0 : neg, 1 : int(neg * 4.957816)})
        X_train_res, y_train_res = rus.fit_resample(X_train_train, y_train_train)
        lgb_ = LGBMClassifier(random_state = 0, **params)
        lgb_.fit(X_train_res, y_train_res)
        pred.append(lgb_.predict_proba(X_train_val)[:, 1])
    bagging_pred = sum(pred) / len(pred)
    smp_rate = sum(y_train_res == 1)/sum(y_train_train == 1)
    pred_proba = np.ones(len(bagging_pred)) - calibrate(bagging_pred, smp_rate)
    def decision_threshold(x):
        y_pred = (pred_proba < x).astype(int)
        return abs(sum(y_pred == 0)/sum(y_train_val == 0) - 0.82)
    result = minimize(decision_threshold, x0 = np.array([0.5]), method='Nelder-Mead')
    print(result["x"].item())
    y_pred = (pred_proba < 0.435507).astype(int)
    print(sum(y_train_val == 0), sum(y_pred == 0))
    pred_Ser.iloc[val_idx] = y_pred
    scores.append(f1_score(y_train_val, y_pred, average="macro"))
tn, fp, fn, tp = confusion_matrix(y_train, pred_Ser).ravel()
print(scores, np.mean(scores))
print(f'TP:{tp}, FP:{fp}, FN:{fn}, TN:{tn}')


# In[ ]:


y_train_for_submit = train["MIS_Status"]
X_train_for_submit = train[lgb_imp_features]
X_test = test[lgb_imp_features]

lgb_submit_params={'num_iterations': 4765, 'learning_rate': 0.009799855546680735, 'max_depth': 8, 'reg_alpha': 34.927535315330154, 'reg_lambda': 116.82707823139178, 'min_child_samples': 572, 'num_leaves': 4137, 'max_bin': 101, 'class_weight':{0: 1.2610662662270764, 1: 1.0}}

print(lgb_submit_params)
pred = []
for i in range(15):
    rus = RandomUnderSampler(random_state=i, replacement=True, sampling_strategy = {0 : neg, 1 : int(neg * 4.957816886784688)})
    X_train_res, y_train_res = rus.fit_resample(X_train_for_submit, y_train_for_submit)
    lgb_ = LGBMClassifier(**lgb_submit_params, random_state = 0)
    lgb_.fit(X_train_res, y_train_res)
    pred.append(lgb_.predict_proba(X_test)[:, 1])
bagging_pred = sum(pred) / len(pred)
smp_rate = sum(y_train_res == 1)/sum(y_train_train == 1)
pred_proba = np.ones(len(bagging_pred)) - calibrate(bagging_pred, smp_rate)
y_pred = (pred_proba < 0.5293).astype(int)
submit[1] = y_pred
submit.to_csv("Desktop/コンペ/第2回 金融データ活用チャレンジ/lgb_tuned_submission.csv", header = None)


# In[ ]:


def calibrate(prob, beta):
    return prob / (prob + (1 - prob) / beta)

y_train = train["MIS_Status"]
X_train = train[lgb_imp_features]

def lgb_objective(trial):
    g = trial.suggest_float('g', 1.0, 8.3)
    trial.set_user_attr('g', g)
    
    params = {'num_iterations':trial.suggest_int('num_iterations', 100, 5000),
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log = True),
              'max_depth':trial.suggest_int('max_depth', 2, 15),
              'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1000, log = True),
              'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1000, log = True),
#              'colsample_bytree':trial.suggest_float('colsample_bytree', 1e-5, 1.0, log = True),
#               'subsample':trial.suggest_float('subsample', 1e-5, 1.0, log = True),
#               'subsample_freq':trial.suggest_int('subsample_freq', 0, 5000),
              'min_child_samples':trial.suggest_int('min_child_samples', 5, 5000),
              "num_leaves": trial.suggest_int("num_leaves", 2, 5000),
              "max_bin": trial.suggest_int("max_bin", 2, 500),
              'class_weight' : {0: trial.suggest_float('class_weight', 1, 3.50), 1: 1.0},
              'verbose' : -1}
    
    scores = []
    thresholds = []
    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    lgb = LGBMClassifier(random_state = 0, early_stopping_round = 10, verbosity=0)
    X_train2, X_stop, y_train2, y_stop = train_test_split(X_train, y_train, test_size=0.2, random_state=0, shuffle = True)
    for train_index, val_index in kf.split(X_train2, y_train2):
        X_train_train, X_train_val = X_train2.iloc[train_index], X_train2.iloc[val_index]
        y_train_train, y_train_val = y_train2.iloc[train_index], y_train2.iloc[val_index]
        pred = []
        neg, pos = np.bincount(y_train_train)
        
        for i in range(15):
            rus = RandomUnderSampler(random_state=i, replacement=True, sampling_strategy = {0 : neg, 1 : int(neg * g)})
            X_train_res, y_train_res = rus.fit_resample(X_train_train, y_train_train)
            lgb.fit(X_train_res, y_train_res, eval_set=[(X_stop, y_stop)])
            pred.append(lgb.predict_proba(X_train_val)[:, 1])
        bagging_pred = sum(pred) / len(pred)
        smp_rate = sum(y_train_res == 1)/sum(y_train_train == 1)
        pred_proba = np.ones(len(bagging_pred)) - calibrate(bagging_pred, smp_rate)
        def decision_threshold(x):
            y_pred = (pred_proba < x).astype(int)
            return abs(sum(y_pred == 0) / sum(y_train_val == 0) - 0.82)
        result = minimize(decision_threshold, x0 = np.array([0.5]), method='Nelder-Mead')
        threshold = result["x"].item()
        thresholds.append(threshold)
        y_pred = (pred_proba < threshold).astype(int)
        score = f1_score(y_train_val, y_pred, average="macro")
        scores.append(score)
    trial.set_user_attr('threshold', np.mean(thresholds))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(lgb_objective, n_trials=100)
best_trial = study.best_trial
lgb_best_g = best_trial.user_attrs['g']
lgb_best_threshold = best_trial.user_attrs['threshold']
lgb_best_params = best_trial.params
print("lgb_best_g:", lgb_best_g)
print("lgb_best_threshold:", lgb_best_threshold)
print("lgb_best_params:", lgb_best_params)


# In[ ]:


def calibrate(prob, beta):
    return prob / (prob + (1 - prob) / beta)

best_score = 0
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
for h in np.arange(1.0, 3.6, 0.1):
    scores = []
    thresholds = []
    for tr_idx, val_idx in kf.split(X_train, y_train):
        X_train_train, X_train_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_train_train, y_train_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        pred = []
        neg, pos = np.bincount(y_train_train)
        for i in range(7):
            rus = RandomUnderSampler(random_state=i, replacement=True)
            X_train_res, y_train_res = rus.fit_resample(X_train_train, y_train_train)
            lgb_ = LGBMClassifier(random_state = 0, class_weight = {0: h, 1: 1.0})
            lgb_.fit(X_train_res, y_train_res)
            pred.append(lgb_.predict_proba(X_train_val)[:, 0])
        bagging_pred = sum(pred) / len(pred)
        def decision_threshold(x):
            y_pred = (bagging_pred < x).astype(int)
            return abs(sum(y_train_val == 0) - sum(y_pred == 0))
        result = minimize(decision_threshold, x0 = np.array([0.5]), method='Nelder-Mead')
        threshold = result["x"].item()
        thresholds.append(threshold)
        y_pred = (bagging_pred < threshold).astype(int)
        score = f1_score(y_train_val, y_pred, average="macro")
        scores.append(score)
    current_score = np.mean(scores)
    if current_score > best_score:
        best_strategy = g
        best_weight = h
        best_score = current_score
        best_threshold = np.mean(thresholds)
    print(h)
print(best_weight, best_score, best_threshold)


# In[ ]:




