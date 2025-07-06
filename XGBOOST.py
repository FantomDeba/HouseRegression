import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as stats
import xgboost as xgb

df = pd.read_csv('processed_houses.csv')
df['log_price'] = np.log1p(df['price'])
df.drop('price', axis=1, inplace=True)
X = df.drop('log_price', axis=1)
y = df['log_price']
X = pd.get_dummies(X, drop_first=True)

to_drop = [
    'Swimming Pool', 'Fitness Centre / GYM', 'Safety', 'Connectivity',
    'Geyser', 'Fan', 'Wardrobe', 'Lifestyle', 'Club house / Community Center',
    'Waste Disposal'
]
for feature in to_drop:
    if feature in X.columns:
        X.drop(feature, axis=1, inplace=True)

from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.02, random_state=42)
outliers = iso.fit_predict(X)
X, y = X[outliers == 1], y[outliers == 1]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xg_reg.fit(X_train, y_train)

y_pred_log = xg_reg.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"XGBoost RMSE: {rmse:.2f}")

residuals = y_test - y_pred_log
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals - XGBoost")
plt.show()

plt.scatter(y_pred_log, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values (log)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted - XGBoost")
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

y_pred_log = xg_reg.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"📊 XGBoost Evaluation Metrics:")
print(f"➡️ MSE        : {mse:.2f}")
print(f"➡️ RMSE       : {rmse:.2f}")
print(f"➡️ R²          : {r2:.4f}")
