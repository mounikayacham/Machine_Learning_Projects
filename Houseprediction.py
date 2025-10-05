import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('C:/Users/Lenovo/Documents/PANDAS/datasets/data.csv')
print(df)
print(df.info())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)
print(df.shape)
print(df.corr)
l=LabelEncoder()
df['street']=l.fit_transform(df['street'])
df['city']=l.fit_transform(df['city'])
df['statezip']=l.fit_transform(df['statezip'])
df['country']=l.fit_transform(df['country'])
X=df.drop(['date','price'],axis=1)
y=df['price']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)
model=LinearRegression()
models={
    'LinearRegressior':LinearRegression(),
    'RandomForestRegressor':RandomForestRegressor(),
    'DecisionTreeRegressor':DecisionTreeRegressor(),
    'SupportVectorRegressor':SVR(),
    'KNeighborsRegressor':KNeighborsRegressor()
}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"\n \n {name} : ")
    print('1.r2_score :',r2_score(y_test,y_pred))
    print('2.mean_squared_error :',mean_squared_error(y_test,y_pred))
    print('3.mean_absolute_error :',mean_absolute_error(y_test,y_pred))
    print('4.root_mean_squared_error :',root_mean_squared_error(y_test,y_pred))
    print('5.mean_absolute_percentage_error :',mean_absolute_error(y_test,y_pred))
    print('6.Symmetric_mean_absolute_percentage_error :',mean_absolute_error(y_test,y_pred))
    print('7.Meadian_bsolute_error :',mean_absolute_error(y_test,y_pred))
    print('8.Explained_variance_score :',explained_variance_score(y_test,y_pred))
    print('9.mean_squared_log_error :',mean_squared_error(y_test,y_pred),'\n \n')

model=RandomForestRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
import joblib
joblib.dump(model,'Houseprediction.pkl')
loaded=joblib.load('Houseprediction.pkl')
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    new_data=X_test[:5]
    print(f"predicton of the {name} :",model.predict(new_data))
    
