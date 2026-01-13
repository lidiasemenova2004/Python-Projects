import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
import xgboost as xgb
from sklearn.linear_model import RidgeCV

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.head())

print(df_train.shape)

print(df_train['SalePrice'].describe())

f, ax = plt.subplots(figsize=(8, 6))
sns.displot(df_train['SalePrice'])

print("Ассиметрия: %f" % df_train['SalePrice'].skew())
print("Эксцесс: %f" % df_train['SalePrice'].kurt())

fig = plt.figure(figsize = (14,8))

fig.add_subplot(1,2,1)
res = stats.probplot(df_train['SalePrice'], plot=plt)

fig.add_subplot(1,2,2)
res = stats.probplot(np.log1p(df_train['SalePrice']), plot=plt)

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
k = 10 # количество коррелирующих признаков, которое мы хотим увидеть
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)

fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)

fig, ax = plt.subplots()
ax.scatter(x = df_train['OverallQual'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)

df_train = df_train.drop(df_train[(df_train['OverallQual'] > 9) & (df_train['SalePrice'] < 220000)].index)
df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index)

df_train.isnull().sum().sort_values(ascending=False).head(20)

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Столбцы", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Общее количество недостающих значений  (%)", fontsize = 20)

Target = 'SalePrice'
df_train.dropna(axis=0, subset=[Target], inplace=True)

all_data = pd.concat([df_train.iloc[:,:-1], df_test],axis=0)

print('У тренировочного датасета {} рядов и {} признаков'.format(df_train.shape[0], df_train.shape[1]))
print('У тестового датасета {} рядов и {} признаков'.format(df_test.shape[0], df_test.shape[1]))
print('Объединённый датасет содержит в себе {} рядов и {} признаков'.format(all_data.shape[0], all_data.shape[1]))

all_data = all_data.drop(columns=['Id'], axis=1)
        
def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False) / len(df)*100, 2)
    temp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return temp.loc[(temp['Total'] > 0)]

missingValuesInfo(df_train)

def HandleMissingValues(df):
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    values = {}
    for a in cat_cols:
        values[a] = 'UNKNOWN'

    for a in num_cols:
        values[a] = df[a].median()
        
    df.fillna(value=values, inplace=True)
    
    
HandleMissingValues(all_data)
all_data.head()

all_data.isnull().sum().sum()

def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == "object"]

def PerformOneHotEncoding(df, columnsToEncode):
    return pd.get_dummies(df, columns=columnsToEncode)

cat_cols = getObjectColumnsList(all_data)
all_data = PerformOneHotEncoding(all_data, cat_cols)
all_data.head()

train_data = all_data.iloc[:1460, :]
test_data = all_data.iloc[1460:, :]
print(df_train.shape)
print(df_test.shape)

ridge_cv = RidgeCV(alphas = (0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
ridge_cv.fit(X, y)
ridge_cv_preds = ridge_cv.predict(test_data)

model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
model_xgb.fit(X, y)
xgb_preds = model_xgb.predict(test_data)

predictions = (ridge_cv_preds + xgb_preds) / 2

submission = {
    'Id': df_test.Id.values,
    'SalePrice': predictions
}
solution = pd.DataFrame(submission)
solution.to_csv('submission.csv',index=False)

plt.show()
