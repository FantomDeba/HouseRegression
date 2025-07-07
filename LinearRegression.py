import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('processed_houses.csv')
df = df.dropna(subset=['price'])
features = ['Maintenance Staff', 'Waste Disposal']
target = 'price'
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
n = len(y)
p = X.shape[1]
rss = np.sum((y - y_pred) ** 2)
sigma2 = np.var(y - y_pred, ddof=p+1)
r2 = r2_score(y, y_pred)
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
aic = n * np.log(rss / n) + 2 * (p + 1)
bic = n * np.log(rss / n) + np.log(n) * (p + 1)
cp = rss / sigma2 - (n - 2 * (p + 1))
print("Performance Metrics:")
print(f"R² Score:       {r2:.4f}")
print(f"Adjusted R²:    {adj_r2:.4f}")
print(f"MAE:            {mae:.4f}")
print(f"MSE:            {mse:.4f}")
print(f"RMSE:           {rmse:.4f}")
print(f"AIC:            {aic:.4f}")
print(f"BIC:            {bic:.4f}")
print(f"Mallow's Cp:    {cp:.4f}")
