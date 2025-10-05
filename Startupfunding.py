import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import *
df=pd.read_csv('C:/Users/Lenovo/Documents/PANDAS/datasets/startup_funding.csv')
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.dtypes)
print(df.isnull())
df['Amount in USD']=df['Amount in USD'].str.replace(',','')
print(df['Amount in USD'].dtypes)
df["Amount in USD"].isna().sum()
df = df.dropna(subset=["Amount in USD"])
# Convert to numeric (will turn invalid strings to NaN)
df["Amount in USD"] = pd.to_numeric(df["Amount in USD"], errors='coerce')
# Now fill NaN with median
df["Amount in USD"]=df["Amount in USD"].fillna(df["Amount in USD"].median())
# remove rows with missing target
print(df.dtypes)
print(df.isnull())
print(df.isnull().sum())
cat_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)   # avoid NaN issues
    df[col] = le.fit_transform(df[col])

# ========================
# 4. Split features & target
# ========================
X = df.drop(["Amount in USD"], axis=1)
y = df["Amount in USD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# 5. Feature scaling
# ========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================
# 6. Train model
# ========================
models={
    'Linear Regression': LinearRegression(),
    'RandomForestRegressor':RandomForestRegressor(),
    'DecisionTreeeRgressior':DecisionTreeRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SupportVectorRegressior':SVR()
}
for name,model in models.items():
    print('\n',name)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("Root meansquared error",root_mean_squared_error(y_test,y_pred))
    print('explained Varience score',explained_variance_score(y_test,y_pred))
model =RandomForestRegressor()
model.fit(X_train,y_train)
import joblib
joblib.dump(model,'Startupfunding.pkl')
loaded=joblib.load('Startupfunding.pkl')
sample_data = X_test[0:5]  # Example sample data
predictions = loaded.predict(sample_data)
print("Predictions on sample data:", predictions)


