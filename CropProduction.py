import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import  SVR
from sklearn.metrics import r2_score,f1_score,accuracy_score,mean_absolute_error,mean_squared_error,roc_auc_score
df=pd.read_csv('C:/Users/Lenovo/Documents/PANDAS/datasets/crop_yield.csv')
print(df.head())
print(df.columns)
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.dtypes)
print(df.shape)
#Encode categorical values
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['Crop']=l.fit_transform(df['Crop'])
df['Season']=l.fit_transform(df['Season'])
df['State']=l.fit_transform(df['State'])
df['Area']=l.fit_transform(df['Area'])
X=df.drop('Production',axis=1)
y=df['Production']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model=LinearRegression()
models={
    'LinearRegression':LinearRegression(),
    'RandomForestRegressor':RandomForestRegressor(),
    'DeciosionTreeRegressor':DecisionTreeRegressor(),
    'SupporVectorRegressor':SVR(),
    'KNeighborsRegressor' :KNeighborsRegressor()
}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f'\n \n{name}')
    print('R2 score :',r2_score(y_test,y_pred))
    print('mean_absolute_error :',mean_absolute_error(y_test,y_pred))
    print('mean_squuared_error:',mean_squared_error(y_test,y_pred))
import joblib
joblib.dump(model, 'Crop_production_model.pkl')
