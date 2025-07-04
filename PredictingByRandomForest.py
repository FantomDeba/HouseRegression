import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
df = pd.read_csv("E:\placement\HouseRegression\processed_houses.csv")
null_counts = df.isnull().sum()
print(df.head())
X = df.drop(columns=['price'])
y = df['price']
y_transformed = np.log1p(df['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestRegressor(100,random_state= 42)
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=5)
print(f"Random Forest: Adjusted R² = {r2_scores.mean():.4f}")