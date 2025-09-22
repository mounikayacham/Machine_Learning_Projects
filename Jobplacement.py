import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
df=pd.read_csv('C:/Users/Lenovo/Documents/PANDAS/datasets/job_placement.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.tail())
print(df.isnull().sum())
print(df.columns)
print(df.dtypes)
print(df.duplicated().sum())
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['name']=l.fit_transform(df['name'])
df['degree']=l.fit_transform(df['degree'])
df['college_name']=l.fit_transform(df['college_name'])
df['placement_status']=l.fit_transform(df['placement_status'])
df['gender']=l.fit_transform(df['gender'])
df['stream']=l.fit_transform(df['stream'])
df['years_of_experience']=df['years_of_experience'].fillna(0)
X=df.drop(['salary'],axis=1)
y=df['salary']
df.columns=df.columns.str.strip()
print(df.isnull().sum())
df=df.fillna(df.mean())
print(df.isnull().sum())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
import pandas as pd
importance = model.feature_importances_
features = pd.DataFrame({"Feature": X.columns, "Importance": importance})
features.sort_values(by="Importance", ascending=False).plot(kind="bar", x="Feature")
plt.title("Feature Importance")
plt.show()
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
joblib.dump(model,'Jobplacement.pkl')
loaded=joblib.load('Jobplacement.pkl')
sample_data = X_test[0:5]  # Example sample data
predictions = loaded.predict(sample_data)
print("Predictions on sample data:", predictions)

