import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("processed_houses.csv")

# Drop missing target values
df = df.dropna(subset=["price"])

# Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# Drop constant features
X = X.loc[:, X.nunique() > 1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Fit model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Residuals from training for Cp calculation
train_pred = model.predict(X_train_scaled)
sigma2 = np.var(y_train - train_pred, ddof=X_train_scaled.shape[1] + 1)

# Evaluation metrics
n = len(y_test)
p = X_train_scaled.shape[1]
rss = np.sum((y_test - y_pred) ** 2)

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
aic = n * np.log(rss / n) + 2 * (p + 1)
bic = n * np.log(rss / n) + np.log(n) * (p + 1)
cp = rss / sigma2 - (n - 2 * (p + 1))

# Print results
print("\n--- Linear Regression Evaluation ---")
print(f"R² Score:       {r2:.4f}")
print(f"Adjusted R²:    {adj_r2:.4f}")
print(f"MAE:            {mae:.4f}")
print(f"MSE:            {mse:.4f}")
print(f"RMSE:           {rmse:.4f}")
print(f"AIC:            {aic:.4f}")
print(f"BIC:            {bic:.4f}")
print(f"Mallow's Cp:    {cp:.4f}")
