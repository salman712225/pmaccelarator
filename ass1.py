# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Load dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

# Step 1: Check column names
print("Column names in the dataset:", df.columns)

# Step 2: Fix column name (use 'last_updated' instead of 'date' or 'lastupdated')
df['last_updated'] = pd.to_datetime(df['last_updated'])
df['year'] = df['last_updated'].dt.year
df['month'] = df['last_updated'].dt.month
df['day'] = df['last_updated'].dt.day

# Step 3: Handle missing values
df = df.ffill()  # Forward fill missing values

# Handle outliers in temperature (example)
Q1 = df['temperature_celsius'].quantile(0.25)
Q3 = df['temperature_celsius'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['temperature_celsius'] >= Q1 - 1.5 * IQR) & (df['temperature_celsius'] <= Q3 + 1.5 * IQR)]

# Exploratory Data Analysis (EDA)
# Basic statistics
print(df.describe())

# Correlation heatmap (only numeric columns)
numeric_columns = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Time series plot for temperature
plt.figure(figsize=(12, 6))
df.set_index('last_updated')['temperature_celsius'].plot()
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

# Model Building: Basic Forecasting
# Prepare data for time series forecasting
ts_data = df.set_index('last_updated')['temperature_celsius'].resample('D').mean()

# Split data into train and test sets
train_size = int(len(ts_data) * 0.8)
train, test = ts_data[:train_size], ts_data[train_size:]

# ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

# Evaluate ARIMA model
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
print(f'ARIMA Model - MAE: {mae}, MSE: {mse}')

# Plot ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Advanced EDA: Anomaly Detection
# Use Isolation Forest to detect anomalies
iso_forest = IsolationForest(contamination=0.05)
df['anomaly'] = iso_forest.fit_predict(df[['temperature_celsius']])
anomalies = df[df['anomaly'] == -1]

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.scatter(df.index, df['temperature_celsius'], label='Normal')
plt.scatter(anomalies.index, anomalies['temperature_celsius'], color='red', label='Anomaly')
plt.title('Anomaly Detection in Temperature')
plt.xlabel('Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Advanced Analysis: Feature Importance
# Train a linear regression model for feature importance
X = df[['humidity', 'wind_kph', 'pressure_mb']]
y = df['temperature_celsius']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Get feature importance
importance = pd.Series(model.coef_, index=X.columns)
importance.plot(kind='bar', title='Feature Importance for Temperature Prediction')
plt.show()

# Advanced Analysis: Geographical Patterns
# Plot temperature by country (example)
geo_data = df.groupby('country')['temperature_celsius'].mean().reset_index()
fig = px.choropleth(geo_data, locations='country', locationmode='country names', color='temperature_celsius',
                    title='Average Temperature by Country')
fig.show()

# Save results to a report
with open("weather_analysis_report.txt", "w") as f:
    f.write("Weather Trend Forecasting Report\n")
    f.write("================================\n")
    f.write(f"ARIMA Model - MAE: {mae}, MSE: {mse}\n")
    f.write("Feature Importance:\n")
    f.write(importance.to_string())
    f.write("\n\nAnomalies Detected:\n")
    f.write(anomalies.to_string())

print("Analysis complete. Report saved as 'weather_analysis_report.txt'.")
