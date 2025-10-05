import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing  import LabelEncoder
df=pd.read_csv('C:/Users/Lenovo/Documents/PANDAS/datasets/improved_disease_dataset.csv')
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.info())
print(df.describe())
print(df.columns)
print(df.dtypes)
print(df.shape)
X=df.drop(['disease'],axis=1)
l=LabelEncoder()
df['disease']=l.fit_transform(df['disease'])
y=df['disease']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(df.dtypes)
print("Accuracy score :",accuracy_score(y_test,y_pred))
import  joblib
joblib.dump(model,'Desease.pkl')
loaded=joblib.load('Desease.pkl')
sample_data = X_test[0:5]  # Example sample data
predictions = loaded.predict(sample_data)
print("Predictions on sample data:", predictions)

