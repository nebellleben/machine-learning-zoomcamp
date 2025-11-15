# 2.1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# import the relevant regressors
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

df = pd.read_csv('StudentsPerformance.csv')

# 2.2

print(df.head())

print(df.columns)

print(df.isna().any())

print(df.dtypes)

to_category = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
df[to_category] = df[to_category].astype('category')

parental_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
df['parental level of education'] = pd.Categorical(df['parental level of education'], categories=parental_order, ordered=True)

print(df.dtypes)

# 2.3

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(df['math score'])
ax[0].set_title('Distribution of math score')

ax[1].hist(df['writing score'])
ax[1].set_title('Distribution of reading score')

ax[2].hist(df['writing score'])
ax[2].set_title('Distribution of writing score')

plt.show()

for i in to_category:
    sns.countplot(data=df, x=i)
    plt.title("Count of Students by Category - %s" % i)
    if i == 'parental level of education':
        plt.xticks(rotation=90)
    plt.show()

sns.heatmap(df[['math score','reading score','writing score']].corr())
plt.suptitle('Correction among math score, reading score and writing score')
plt.show()

sns.pairplot(df[['math score','reading score','writing score']])
plt.suptitle('Interaction among math score, reading score and writing score')
plt.show()

# 2.4

for cat in to_category:
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    for i,d in enumerate(['math score', 'reading score', 'writing score']):
        sns.boxplot(x=cat, y=d, data=df, ax=ax[i])
    plt.suptitle(f'Effect of {cat} on Scores')
    if cat == 'parental level of education':
        plt.xticks(rotation=90)
    plt.show()


# Split the dataset

df_all_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_all_train, test_size=0.25, random_state=42)

y_train = []
y_val = []
y_all_train = []
y_test = []

eval_cols = ['math score', 'reading score', 'writing score']

for e in eval_cols:
    y_train.append(df_train[e].values)
    y_val.append(df_val[e].values)
    y_all_train.append(df_all_train[e].values)
    y_test.append(df_test[e].values)

    del df_train[e]
    del df_val[e]
    del df_all_train[e]
    del df_test[e]

# Use Data Vectorizer
dv = DictVectorizer(sparse=True)

train_dict = df_train.to_dict(orient='records')
all_train_dict = df_all_train.to_dict(orient='records')
val_dict = df_val.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')

X_train = dv.fit_transform(train_dict)
X_all_train = dv.transform(all_train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)

# columns:
print(dv.get_feature_names_out())


# Initialise models and predicted results

models = {}
predictions = {}

# Linear Regression (Lasso)

# For Math, Reading, Writing [0,1,2]:
#lr = []
#pred_lr = []

models['lr'] = []
predictions['lr'] = []

lasso_alpha = [0.001, 0.01, 0.1, 1, 10]

for la in lasso_alpha:
    for i in range(3):
        models['lr'].append(Lasso(alpha=la))
        models['lr'][i].fit(X_train, y_train[i])
        y_pred = models['lr'][i].predict(X_val)
        predictions['lr'].append(y_pred)
    print(f'Alpha: {la}, RMSE: {(mean_squared_error(y_val[0], predictions['lr'][0])+mean_squared_error(y_val[1], predictions['lr'][1])+mean_squared_error(y_val[2], predictions['lr'][2]))**0.5}')

# Decision Tree Regressor

# For Math, Reading, Writing [0,1,2]:

dt_max_depths = [1,5,10,15]
dt_min_samples_split = [2,5,7,10]

models['dt'] = []
predictions['dt'] = []
for mss in dt_min_samples_split:
    for md in dt_max_depths:
        for i in range(3):
            models['dt'].append(DecisionTreeRegressor(max_depth=md, min_samples_split=mss))
            models['dt'][i].fit(X_train, y_train[i])
            predictions['dt'].append(models['dt'][i].predict(X_val))
        print(f'max_depth: {md}, min_samples_split: {mss}, RMSE: {(mean_squared_error(y_val[0], predictions['dt'][0])+mean_squared_error(y_val[1], predictions['dt'][1])+mean_squared_error(y_val[2], predictions['dt'][2]))**0.5}')

# Tune max_depth, min_samples_split

# Random Forest Regressor

# For Math, Reading, Writing [0,1,2]:

rf_n_estimators = [1,5,10,15]
rf_max_depth = [1,5,10,15]

models['rf'] = []
predictions['rf'] = []
for ne in rf_n_estimators:
    for md in rf_max_depth:
        for i in range(3):
            models['rf'].append(RandomForestRegressor(n_estimators = ne, max_depth = md))
            models['rf'][i].fit(X_train, y_train[i])
            predictions['rf'].append(models['rf'][i].predict(X_val))

        print(f'max_depth: {md}, n_estimators: {ne}, RMSE: {(mean_squared_error(y_val[0], predictions['rf'][0])+mean_squared_error(y_val[1], predictions['rf'][1])+mean_squared_error(y_val[2], predictions['rf'][2]))**0.5}')

# tune n_estimators and max_depth

# XGBoost

etas = [1, 0.5, 0.3, 0.1, 0.05]
features = list(dv.get_feature_names_out())

models['xgb'] = []
predictions['xgb'] = []

for eta in etas:

    score = 0
    for i in range(3):
        
        dtrain = xgb.DMatrix(X_train, label=y_train[i], feature_names=features)
        dval = xgb.DMatrix(X_val, label=y_val[i], feature_names=features)

    
        xgb_params = {
            'eta': eta,
            'max_depth': 20,
            'min_child_weight': 1,
            
            'objective': 'reg:squarederror',
            'nthread': 8,
            
            'seed': 1,
            'verbosity': 1,
        }

        #model = xgb.train(
        #    xgb_params,
        #    dtrain,
        #    evals=watchlist,
        #    verbose_eval=0,
        #    num_boost_round=100,
        #    evals_result=evals_result
        #)

        model = xgb.train(xgb_params,dtrain=dtrain,num_boost_round=100)

        models['xgb'].append(model)

        predictions['xgb'].append(model.predict(dval))
        score += mean_squared_error(y_val[i], predictions['xgb'][i])

    score = score**0.5
    print(f'eta:{eta}, RMSE:%.3f' % score)

types = ['Math', 'Reading', 'Writing']
methods = {'lr':'Linear Regression','dt':'Decision Tree','rf':'Random Forest','xgb':'XGBoost'}

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12, 10), sharex=True, sharey=True)

for index, (m,n) in enumerate(methods.items()):
    for i,name in enumerate(types):
        #y_var = globals()[f'pred_{m}'][i]
        y_var = predictions[m][i]
        sns.scatterplot(x=y_val[i], y=y_var, ax=ax[index,i])
        min_val = min(y_val[i].min(), y_var.min())
        max_val = max(y_val[i].max(), y_var.max())
        line = np.linspace(min_val, max_val, 2)
        ax[index,i].plot(line, line, color='red')
        ax[index,i].set_title(f'{n} - {name}')

fig.tight_layout(pad=2.0) 
fig.suptitle("Actual vs Predicted Scores", fontsize=16, y=1.02, fontweight='bold')

plt.show()


import pickle

# Save the model - math, reading, writing
with open('score_predict_model_lr.pkl', 'wb') as f:
    pickle.dump((dv, models['lr'][0], models['lr'][1], models['lr'][2]), f)