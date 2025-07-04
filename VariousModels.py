import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import numpy as np
df = pd.read_csv("E:\placement\HouseRegression\processed_houses.csv")
null_counts = df.isnull().sum()
print(df.head())
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid = GridSearchCV(ridge, params, scoring='r2', cv=5)
grid.fit(X_train_scaled, y_train)

best_ridge = grid.best_estimator_
print(f"Best alpha: {grid.best_params_['alpha']}")

models = {
    'Ridge': Ridge(alpha=100),
    'Lasso': Lasso(alpha=100,fit_intercept= True),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

from sklearn.metrics import make_scorer, mean_absolute_percentage_error
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# Evaluate models
for name, model in models.items():
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=5)
    mape_scores = cross_val_score(model, X, y, scoring=mape_scorer, cv=5)

    print(f"{name}: Adjusted R² = {r2_scores.mean():.4f}")
    print(f"{name}: MAPE = {-mape_scores.mean():.4f}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dense(1)  # Output layer (regression)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=1)
y_pred = model.predict(X_test_scaled).flatten()
from sklearn.metrics import r2_score, mean_squared_error

print("R² Score:", r2_score(y_test, y_pred))

y_transformed = df['log_price'] = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid = GridSearchCV(ridge, params, scoring='r2', cv=5)
grid.fit(X_train_scaled, y_train)

best_ridge = grid.best_estimator_
print(f"Best alpha: {grid.best_params_['alpha']}")

models = {
    'Ridge': Ridge(alpha=100),
    'Lasso': Lasso(alpha=100,fit_intercept= True),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

from sklearn.metrics import make_scorer, mean_absolute_percentage_error
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# Evaluate models
for name, model in models.items():
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=5)
    mape_scores = cross_val_score(model, X, y, scoring=mape_scorer, cv=5)

    print(f"{name}: Adjusted R² = {r2_scores.mean():.4f}")
    print(f"{name}: MAPE = {-mape_scores.mean():.4f}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dense(1)  # Output layer (regression)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=1)
y_pred = model.predict(X_test_scaled).flatten()
from sklearn.metrics import r2_score, mean_squared_error

print("R² Score:", r2_score(y_test, y_pred))

# we select random forest from here and apply the rest of the portions with that.

