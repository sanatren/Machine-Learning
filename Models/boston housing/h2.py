import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('housing.csv')

# Feature selection based on correlation
corr_matrix = df.corr()
relevant_features = corr_matrix['MEDV'].sort_values(ascending=False).index[:3]  # Select top correlated features
X = df[relevant_features]
Y = df['MEDV']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Models to try
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge()
}

# Hyperparameters to tune
params = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.1, 1, 10, 100]}
}

# GridSearchCV and model fitting
for name, model in models.items():
    grid_search = GridSearchCV(model, params[name], scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(x_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation R2 score for {name}: {best_score}")

    # Evaluate on test set
    y_pred = best_model.predict(x_test_scaled)
    test_r2 = r2_score(y_test, y_pred)
    print(f"Test R2 score for {name}: {test_r2}")
