#===========================================================
#      Stock Price Prediction using Machine Learning
#
# Build a model to predict future stock prices based on historical
# stock data, including features like opening price, closing
# price, high, low, and trading volume. Task: pre-process the
# data, choose a suitable model (Linear Regression / Random
# Forest Regressor), train it, and evaluate prediction accuracy.
#
# NOTE: This is a REGRESSION problem (predicting a continuous
# price), not a classification problem. All models/metrics below
# are regression-appropriate. Classification metrics such as
# confusion matrix, accuracy, precision, recall, and F1-score do
# NOT apply to this task -- they are used for the Titanic
# Survival Classification task instead.
#=============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# -----------------------------------------------------------
# 1. LOAD DATA & INITIAL SUMMARY
# -----------------------------------------------------------

df = pd.read_csv(r'c:\Users\NEW LAPTOP CITY\.vscode\Internship Projects\2. Project 1\Task 2\NFLX.csv')   # <-- update this path if running locally

print("="*70)
print(" 1. INITIAL DATASET OVERVIEW")
print("="*70)
print(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}\n")

print("--- Sample Data (First 5 Rows) ---")
print(df.head())

print("\n--- Data Structure & Non-Null Counts ---")
print(df.info())

print("\n--- Numerical Summary Statistics ---")
print(df.describe())

# -----------------------------------------------------------
# 1b. FULL DESCRIPTIVE STATISTICS (expanded)
# -----------------------------------------------------------
print("\n" + "="*70)
print(" 1b. EXPANDED DESCRIPTIVE STATISTICS")
print("="*70)

numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
desc_stats = df[numeric_cols].agg(
    ['mean', 'median', 'std', 'var', 'min', 'max', 'skew', 'kurt']
).T
desc_stats.columns = ['Mean', 'Median', 'Std Dev', 'Variance',
                       'Min', 'Max', 'Skewness', 'Kurtosis']
print(desc_stats.round(3))
desc_stats.round(3).to_csv('descriptive_statistics.csv')

# -----------------------------------------------------------
# 2. PRE-PROCESSING
# -----------------------------------------------------------
print("="*70)
print(" 2. PRE-PROCESSING")
print("="*70)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

if 'Adj Close' in df.columns and (df['Close'] == df['Adj Close']).all():
    df = df.drop(columns=['Adj Close'])

# -----------------------------------------------------------
# 3. FEATURE ENGINEERING & TARGET FORMULATION
# -----------------------------------------------------------
print("="*70)
print(" 3. FEATURE ENGINEERING")
print("="*70)

# Stationarized Target: Next day's percentage return
df['Target_Return'] = df['Close'].pct_change().shift(-1)

# Lagged Features (using strictly past data)
for lag in [1, 2, 3, 5]:
    df[f'Close_lag{lag}'] = df['Close'].shift(lag)
    df[f'Volume_lag{lag}'] = df['Volume'].shift(lag)

# Moving Averages based on past Close prices
df['MA5'] = df['Close'].shift(1).rolling(window=5).mean()
df['MA10'] = df['Close'].shift(1).rolling(window=10).mean()

# Historical Price Dynamics & Returns
df['Daily_Range_Lag1'] = (df['High'] - df['Low']).shift(1)
df['Daily_Return_Lag1'] = df['Close'].pct_change().shift(1)

# Drop initial missing values caused by lagging and rolling features
df = df.dropna().reset_index(drop=True)

# -----------------------------------------------------------
# 3b. CORRELATION HEATMAP
# -----------------------------------------------------------
print("="*70)
print(" 3b. CORRELATION HEATMAP")
print("="*70)

corr_cols = [
    'Close', 'Volume', 'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5',
    'Volume_lag1', 'Volume_lag2', 'Volume_lag3', 'Volume_lag5',
    'MA5', 'MA10', 'Daily_Range_Lag1', 'Daily_Return_Lag1', 'Target_Return'
]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Heatmap (NFLX Stock Data)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("Correlation heatmap saved -> correlation_heatmap.png")

# -----------------------------------------------------------
# 4. TRAIN / TEST SPLIT (CHRONOLOGICAL)
# -----------------------------------------------------------
print("="*70)
print(" 4. TRAIN / TEST SPLIT")
print("="*70)

df_model = df.dropna(subset=['Target_Return']).reset_index(drop=True)

feature_cols = [
    'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5',
    'Volume_lag1', 'Volume_lag2', 'Volume_lag3', 'Volume_lag5',
    'MA5', 'MA10', 'Daily_Range_Lag1', 'Daily_Return_Lag1'
]

X = df_model[feature_cols]
y = df_model['Target_Return']

split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

actual_close_test = df_model['Close'].iloc[split_idx:]
actual_next_close = df_model['Close'].shift(-1).iloc[split_idx:]
dates_test = df_model['Date'].iloc[split_idx:]

valid_mask = actual_next_close.notna()
actual_close_test = actual_close_test[valid_mask]
actual_next_close = actual_next_close[valid_mask]
dates_test = dates_test[valid_mask]
X_test = X_test[valid_mask]
y_test = y_test[valid_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# 5. MODEL TRAINING & PREDICTION
# -----------------------------------------------------------
print("="*70)
print(" 5. MODEL TRAINING")
print("="*70)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred_returns = lr_model.predict(X_test_scaled)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred_returns = rf_model.predict(X_test)

lr_preds = actual_close_test * (1 + lr_pred_returns)
rf_preds = actual_close_test * (1 + rf_pred_returns)
naive_preds = actual_close_test.values

# -----------------------------------------------------------
# 6. EVALUATION (REGRESSION METRICS ONLY -- see note at top)
# -----------------------------------------------------------
print("="*70)
print(" 6. MODEL EVALUATION (ON DOLLAR PRICES)")
print("="*70)
print("NOTE: Accuracy / Precision / Recall / F1 / Confusion Matrix are")
print("classification metrics and are NOT applicable to this regression")
print("task. MAE, RMSE, and R2 are the correct evaluation metrics here.\n")

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"MAE  (Mean Absolute Error):     ${mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
    print(f"R2 Score:                       {r2:.4f}")
    return mae, rmse, r2

naive_metrics = evaluate("Naive Baseline (Persistence)", actual_next_close, naive_preds)
lr_metrics = evaluate("Linear Regression (Return-based)", actual_next_close, lr_preds)
rf_metrics = evaluate("Random Forest (Return-based)", actual_next_close, rf_preds)

# -----------------------------------------------------------
# 7. COMPARISON & VISUALIZATION
# -----------------------------------------------------------
print("\n" + "="*70)
print(" SUMMARY COMPARISON")
print("="*70)
results = pd.DataFrame({
    'Model': ['Naive Baseline', 'Linear Regression', 'Random Forest'],
    'MAE ($)': [naive_metrics[0], lr_metrics[0], rf_metrics[0]],
    'RMSE ($)': [naive_metrics[1], lr_metrics[1], rf_metrics[1]],
    'R2 Score': [naive_metrics[2], lr_metrics[2], rf_metrics[2]]
})
print(results.to_string(index=False))
results.to_csv('model_comparison_results.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(dates_test.values, actual_next_close.values, label='Actual Close Price', color='black', linewidth=1.5)
plt.plot(dates_test.values, lr_preds.values, label='Linear Regression Prediction', linestyle='--', alpha=0.8)
plt.plot(dates_test.values, rf_preds.values, label='Random Forest Prediction', linestyle='--', alpha=0.8)
plt.title('NFLX Stock Price Prediction: Stationary Model Performance')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('prediction_comparison_corrected.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 6))
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
sns.barplot(x=importances.values, y=importances.index)
plt.title('Random Forest Feature Importance (Predicting Return)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_corrected.png', dpi=150)
plt.show()