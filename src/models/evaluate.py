import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, max_error
)

# Load the trained model
xgb = joblib.load('../../models/xgb_model.pkl')

X_test = pd.read_csv('../../data/processed/X_test.csv')
y_test = pd.read_csv('../../data/processed/y_test.csv')
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]  # Convert DataFrame to Series if single column

y_pred_xgb = xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_xgb)
max_err = max_error(y_test, y_pred_xgb)

print("----- XGBoost Regressor Evaluation -----")
print(f"MAE  (Mean Absolute Error): {mae:.4f}")
print(f"MSE  (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R²   (R-squared Score): {r2:.4f}")
print(f"Max Error: {max_err:.4f}")
