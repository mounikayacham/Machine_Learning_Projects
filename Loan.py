import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,r2_score,classification_report
from sklearn.preprocessing  import OneHotEncoder,LabelEncoder
data=pd.read_csv("C:/Users/Lenovo/Documents/PANDAS/datasets/loan_approval_dataset.csv")
print(data)
print(data.head())
print(data.info)
print(data.describe())
print(data.isnull())
print(data.duplicated().sum())
print(data.isnull().sum())
print(data.shape)
print(data.dtypes)
print(data.columns)
print(data.info)
data.columns = data.columns.str.strip()
print(data.columns)
l=LabelEncoder()
data['loan_status']=l.fit_transform(data['loan_status'])
print(data['loan_status'].head())
X = data.drop(['loan_status','education','self_employed'], axis=1)
y = data['loan_status']
print(X)
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
models={
    'Logistic Regression':LogisticRegression(max_iter=1000),
    'Random Forest':RandomForestClassifier(),
    'Decision Tree':DecisionTreeClassifier(),
    'Gradient Boosting':GradientBoostingClassifier(),
    'Support Vector Machine':SVC()
}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred,average='weighted',zero_division=1)
    f1=f1_score(y_test,y_pred,average='weighted',zero_division=1)
    r2=r2_score(y_test,y_pred)
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, R2 Score: {r2:.4f}")
    print(classification_report(y_test,y_pred,zero_division=1))
    print("\n")# Save the best model
best_model = max(models.items(), key=lambda x: accuracy_score(y_test, x[1].predict(X_test)))[1]
import joblib
joblib.dump(best_model, 'best_loan_approval_model.pkl')
print("Best model saved as 'best_loan_approval_model.pkl'")
# Load the model
loaded_model = joblib.load('best_loan_approval_model.pkl')
# Make predictions with the loaded model
sample_data = X_test.iloc[0:5]  # Example sample data
predictions = loaded_model.predict(sample_data)
print("Predictions on sample data:", predictions)

