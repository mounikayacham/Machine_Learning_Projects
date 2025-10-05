import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import *
data=pd.read_csv("C:/Users/Lenovo/Documents/PANDAS/datasets/medical_costs.csv")
l=LabelEncoder()
data['Sex']=l.fit_transform(data['Sex'])
data['Smoker']=l.fit_transform(data['Smoker'])
data['Region']=l.fit_transform(data['Region'])
print(data['Sex'].head())
print(data.tail())
print(data.shape)
print(data.info)
print(data.describe())
print(data.isnull().sum())
print(data.dtypes)
X=data.drop('Medical Cost',axis=1)
y=data['Medical Cost']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
models={
    'RandomForestRegressor':RandomForestRegressor(),
    'DecisionTreeRegressor':DecisionTreeRegressor(),
    'SupportVectorMachine':SVR(),
    'KNeighborsRegressor':KNeighborsRegressor(),
    'LinearRegression':LinearRegression()
}
for name,model in models.items():

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f'\n\n{name}:\n')
    print("1.mean_Squared_error:\t",mean_absolute_error(y_test,y_pred))
    print('2.r2_score:\t',r2_score(y_test,y_pred))
    print('3.mean_absolute_error:\t',mean_absolute_error(y_test,y_pred))
    print('4.root_mean_squred_error:\t',root_mean_squared_error(y_test,y_pred))
    print('5.symmetric_mean_absolute_error:\t',explained_variance_score(y_test,y_pred))
import joblib
model=LinearRegression()
model.fit(X_train,y_train)
joblib.dump(model,'Medicalcost.pkl')
loaded=joblib.load('Medicalcost.pkl')
sample_data = X_test[0:5]  # Example sample data
predictions = loaded.predict(sample_data)
print("Predictions on sample data:", predictions)
