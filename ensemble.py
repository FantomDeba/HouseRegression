import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
df = pd.read_csv('processed_houses.csv')
df['log_price'] = np.log1p(df['price'])
df.drop('price', axis=1, inplace=True)

X = pd.get_dummies(df.drop('log_price', axis=1), drop_first=True)
y = df['log_price']

to_drop = [
    'Swimming Pool', 'Fitness Centre / GYM', 'Safety', 'Connectivity',
    'Geyser', 'Fan', 'Wardrobe', 'Lifestyle', 'Club house / Community Center',
    'Waste Disposal'
]
X = X.drop(columns=[col for col in to_drop if col in X.columns], errors='ignore')
from sklearn.ensemble import IsolationForest
mask = IsolationForest(contamination=0.02, random_state=42).fit_predict(X)
X, y = X[mask == 1], y[mask == 1]
X_scaled = RobustScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
base_models = [
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 10))),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42))
]
meta_model = LassoCV(alphas=np.logspace(-4, 1, 20), random_state=42)

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,  
    cv=5,
    n_jobs=-1
)
stacked_model.fit(X_train, y_train)
y_pred_log = stacked_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f" Stacked Ensemble Evaluation:")
print(f"RMSE       : {rmse:.2f}")
print(f" R²          : {r2:.4f}")

import matplotlib.pyplot as plt
import scipy.stats as stats
residuals = y_test - y_pred_log
fitted_vals = y_pred_log
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals - Stacked Model")
plt.grid()
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(fitted_vals, residuals, alpha=0.5, edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values (log scale)")
plt.ylabel("Residuals (log scale)")
plt.title("Residuals vs Fitted Values - Stacked Model")
plt.grid()
plt.show()