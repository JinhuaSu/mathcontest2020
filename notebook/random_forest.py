from numpy import * 
import numpy as np
import pandas as pd
from math import log
import operator


# 提高随机森林的精度：指标选取，数据处理，参数选择

# 数据处理
df = pd.read_csv("../data/zhibiao3.csv")

# L2正则法
for i in ["shangyoutui", "xiayoutui"]:
    df[i] = (max(df[i]) - df[i])/sqrt(sum(df[i]*df[i]))
for i in ["company", "revenue", "profit", "profit_rate", "shangyouji", "xiayouji","shangyoudingdan","xiayoudingdan"]:
    df[i] = df[i]/sqrt(sum(df[i]*df[i]))
    
    
data = df.values[:,1:].tolist()
data_full= data[:]
labels=df.columns.values[1:-1].tolist()
labels_full=labels[:]

# 随机森林
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
x, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 7)
feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=130, max_depth=5, oob_score=True, min_samples_leaf = 5, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
ypred = forest.predict(x_test)

# 混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
mat = confusion_matrix(y_test, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=["A","B","C","D"], yticklabels=["A","B","C","D"])
print(accuracy_score(y_test, ypred))




# 调参
from sklearn.model_selection import GridSearchCV
# 建立n_estimators为45的随机森林
rfc = RandomForestClassifier(max_depth=3)

# 用网格搜索调整max_depth
param_grid = {'max_features':np.arange(1,20),'n_estimators':np.arange(40,60,10),'random_state':np.arange(30,150,20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(x_train, y_train)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)