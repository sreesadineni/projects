import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
input=pd.read_csv("Remote_User/data1.csv")
X = input.drop(['fraud'],axis=1)
y = input['fraud']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True,stratify=y)
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train[:300000].values, y_train.values.ravel()[:300000])
y_pred = clf.predict(X_test.values)
# print(clf.predict([[4,1,18660]]))