import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../data/features/data_cleaned.csv')

X = df.drop(columns=['track_popularity'])
y = df['track_popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_linear = model.predict(X_test)

# Random Forest Model
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)
y_pred_rf = forest.predict(X_test)

# XGBoost Model
xgb_params = {'max_depth': 20,
              'eta':0.1,
              'subsample': 0.8,
              'colsample_bytree': 0.6,
              'seed': 2018,
              'objective': 'reg:squarederror'}
xgb = XGBRegressor(**xgb_params)
xgb.fit(X, y)
y_pred_xgb = xgb.predict(X_test)

# Save the training models
joblib.dump(model, '../../models/regression_model.pkl')
joblib.dump(forest, '../../models/forest_model.pkl')
joblib.dump(xgb, '../../models/xgb_model.pkl')
print("Models saved successfully!")
