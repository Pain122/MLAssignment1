import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Data Preprocessing
df_train_clf = pd.read_csv('Classification/train_data.csv')
df_test_clf = pd.read_csv('Classification/test_data.csv')


# Categorical encoding
def one_hot_encode(df: pd.DataFrame, f_name: str):
    ohe = OneHotEncoder(sparse=False).fit_transform(df[f_name].to_numpy().reshape(-1, 1))
    encoded = pd.DataFrame(ohe, columns=[f"{f_name}{i}" for i in range(ohe.shape[1])])
    return pd.concat([df.drop([f_name], axis=1), encoded], axis=1)


f = 'auto_bitrate_state'
ohe_train_clf = one_hot_encode(df_train_clf, f)
ohe_test_clf = one_hot_encode(df_test_clf, f)


def label_encode(df: pd.DataFrame, f_name: str):
    label = LabelEncoder().fit_transform(df[f_name])
    res = df.copy()
    res[f_name] = label
    return res


f = 'auto_fec_state'
prep_train_clf = label_encode(ohe_train_clf, f)
prep_test_clf = label_encode(ohe_test_clf, f)


def scale(df: pd.DataFrame):
    return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)


X_train_clf, X_test_clf = scale(prep_train_clf.drop(['stream_quality'], axis=1)), scale(
    prep_test_clf.drop(['stream_quality'], axis=1))
y_train_clf, y_test_clf = prep_train_clf['stream_quality'], prep_test_clf['stream_quality']

# Feature scaling
df_train_reg: pd.DataFrame = pd.read_csv('Regression/train_data.csv')
df_test_reg: pd.DataFrame = pd.read_csv('Regression/test_data.csv')
reg_train_sc = scale(df_train_reg)
reg_test_sc = scale(df_test_reg)
X_train_reg, X_test_reg = reg_train_sc.drop(['target'], axis=1), reg_test_sc.drop(['target'], axis=1)
y_train_reg, y_test_reg = reg_train_sc['target'], reg_test_sc['target']
y_train_clf, y_test_clf = y_train_clf.astype(int), y_test_clf.astype(int)

# Dimensionality reduction (exact reasoning why I did it this way can be seen in ipynb)
X_train_reg, X_test_reg = X_train_reg.drop(['rtt_std'], axis=1), X_test_reg.drop(['rtt_std'], axis=1)
X_train_clf, X_test_clf = X_train_clf.drop(['auto_bitrate_state0'], axis=1), X_test_clf.drop(['auto_bitrate_state0'],
                                                                                             axis=1)

# Data visualization can be seen in the ipynb

# Regression model
model1 = LinearRegression()
model2 = Ridge()
model1.fit(X_train_reg, y_train_reg)
model2.fit(X_train_reg, y_train_reg)

y_pred_reg1 = model1.predict(X_test_reg)
y_pred_reg2 = model2.predict(X_test_reg)

# Finding best polynomial regression model
res = []

for i in range(2, 5):
    poly = make_pipeline(PolynomialFeatures(i), LinearRegression())
    poly.fit(X_train_reg, y_train_reg)
    y_pred_reg3 = poly.predict(X_test_reg)
    res.append(mean_squared_error(y_test_reg, y_pred_reg3))
print('----------------------------------------------------')
print(f'Results of Polynomial Regression: {res}')

# Logistic Regression model

logistic = LogisticRegression(penalty='l2', max_iter=120000)
logistic.fit(X_train_clf, y_train_clf)
y_pred_clf = logistic.predict(X_test_clf)

# Results evaluation
# Regression

model1 = LinearRegression()
model2 = Ridge()
model3 = make_pipeline(PolynomialFeatures(2), LinearRegression())

m1_cross = cross_val_score(model1, X_train_reg, y_train_reg, cv=5)
m2_cross = cross_val_score(model2, X_train_reg, y_train_reg, cv=5)
m3_cross = cross_val_score(model3, X_train_reg, y_train_reg, cv=5)

print('----------------------------------------------------')

print(f'Model 1 Cross Validation mean: {np.mean(m1_cross)}, std: {np.std(m1_cross)}')
print(f'Model 2 Cross Validation mean: {np.mean(m2_cross)}, std: {np.std(m2_cross)}')
print(f'Model 3 Cross Validation mean: {np.mean(m3_cross)}, std: {np.std(m3_cross)}')
print('----------------------------------------------------')
print(f'Model 1 (Linear) - MSE:{mean_squared_error(y_test_reg, y_pred_reg1)}')
print(f'Model 2 (Linear + Ridge) - MSE:{mean_squared_error(y_test_reg, y_pred_reg2)}')
print(f'Model 3 (Polynomial with degree {np.argmin(res) + 2}) - MSE:{res[np.argmin(res)]}')

# Classification

logistic = LogisticRegression(penalty='l2', max_iter=120000)

log_cross = cross_val_score(logistic, X_train_clf, y_train_clf, cv=5)
print('----------------------------------------------------')
print(f'Logistic Regression Cross Validation mean: {np.mean(log_cross)}, std: {np.std(log_cross)}')
print('----------------------------------------------------')
print(classification_report(y_test_clf, y_pred_clf))

# Outliers detection
# Finding outliers
X_train_clf_o = X_train_clf.copy()
X_train_clf_o['target'] = y_train_clf
z = np.abs(stats.zscore(X_train_clf))

X_train_clf_o = X_train_clf_o[(z < 3).all(axis=1)]
print('----------------------------------------------------')
print(f'Removed Outliers: {X_train_clf.shape[0] - X_train_clf_o.shape[0]}')

X_train_clf_o, y_train_clf_o = X_train_clf_o.drop('target', axis=1), X_train_clf_o['target']

smt = RandomOverSampler(sampling_strategy='minority')
X_res, y_res = smt.fit_resample(X_train_clf_o, y_train_clf_o)

logistic = LogisticRegression(penalty='l2', max_iter=120000)

logistic.fit(X_res, y_res)
print('----------------------------------------------------')
print(f'Number of classes: 0 - {sum(y_res == 0)}, 1 - {sum(y_res == 1)}')

log_cross = cross_val_score(logistic, X_res, y_res, cv=5)
print('----------------------------------------------------')
print(f'Logistic Regression with sampling Cross Validation mean: {np.mean(log_cross)}, std: {np.std(log_cross)}')
print('----------------------------------------------------')
y_pred = logistic.predict(X_test_clf)
print('After outliers detection')
print(classification_report(y_test_clf, y_pred))
print('Before outliers detection')
print(classification_report(y_test_clf, y_pred_clf))
