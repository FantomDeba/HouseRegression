import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats


df = pd.read_csv("E:/placement/HouseRegression/processed_houses.csv")
X = df.drop(columns=['price'])
y = np.log1p(df['price'])  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)


y_pred = rf.predict(X_test_scaled)
residuals = y_test - y_pred


plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.grid(True)
plt.show()


shapiro_stat, shapiro_pval = stats.shapiro(residuals)
print("Shapiro-Wilk Test Statistic:", shapiro_stat)
print("p-value:", shapiro_pval)


from scipy.stats import zscore
import numpy as np
import pandas as pd


y_train_z = zscore(y_train)
outlier_mask = np.abs(y_train_z) > 3


print(f"Number of outliers in y_train based on Z-score: {np.sum(outlier_mask)}")
print(y_train[outlier_mask])

X_train_df = pd.DataFrame(X_train_scaled)
z_scores = np.abs(zscore(X_train_df))
outliers = (z_scores > 3).any(axis=1)
print(f"Total feature-based outliers (Z > 3): {np.sum(outliers)}")


from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
outlier_flags = iso.fit_predict(X_train_scaled)

print(f"Detected outliers based on IsolationForest : {(outlier_flags == -1).sum()}")