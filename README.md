Weather Trend Forecasting
Project Overview
This project analyzes the Global Weather Repository dataset to forecast future weather trends. The dataset contains daily weather information for cities worldwide, including features such as temperature, humidity, wind speed, and air quality. The analysis includes:

Data cleaning and preprocessing.

Exploratory data analysis (EDA).

Time series forecasting using ARIMA.

Advanced analyses such as anomaly detection, feature importance, and geographical pattern visualization.

Dataset
The dataset used in this project is available on Kaggle:
Global Weather Repository

Key Features
last_updated: Timestamp of the weather data.

temperature_celsius: Temperature in Celsius.

humidity: Humidity percentage.

wind_kph: Wind speed in kilometers per hour.

pressure_mb: Atmospheric pressure in millibars.

country: Country where the weather data was recorded.

Project Structure
The project is organized as follows:

  
weather-trend-forecasting/
├── data/
│   └── Global_Weather_Repository.csv       # Dataset
├── notebooks/
│   └── weather_forecasting.ipynb          # Jupyter Notebook for analysis
├── scripts/
│   └── weather_forecasting.py             # Python script for analysis
├── outputs/
│   ├── correlation_heatmap.png            # Correlation heatmap
│   ├── temperature_trend.png              # Temperature over time plot
│   ├── arima_forecast.png                 # ARIMA forecast plot
│   ├── anomaly_detection.png              # Anomaly detection plot
│   ├── feature_importance.png             # Feature importance plot
│   └── geographical_patterns.png          # Geographical patterns plot
├── reports/
│   └── weather_analysis_report.txt        # Summary report
└── README.md                              # Project overview
Requirements
To run the code, you need the following Python libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Statsmodels

Plotly

You can install the required libraries using the following command:

 
  
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels plotly
How to Run
Clone the Repository:


git clone https://github.com/your-username/weather-trend-forecasting.git
cd weather-trend-forecasting
Download the Dataset:

Download the dataset from Kaggle.

Place the Global_Weather_Repository.csv file in the data/ folder.

Run the Script:

To run the Python script:

python scripts/weather_forecasting.py
To use the Jupyter Notebook:

 
  
jupyter notebook notebooks/weather_forecasting.ipynb
View Results:

Visualizations and outputs will be saved in the outputs/ folder.

A summary report will be saved as reports/weather_analysis_report.txt.

Results
Key Findings
Temperature Trends:

Seasonal patterns were observed in temperature data, with clear peaks and dips corresponding to summer and winter.

Anomalies:

Anomalies in temperature data may indicate extreme weather events or data collection errors.

Feature Importance:

Humidity is the most significant factor affecting temperature, followed by wind speed and pressure.

Geographical Patterns:

Temperature varies significantly across countries, with equatorial regions being the warmest.

Visualizations
Correlation Heatmap:
Correlation Heatmap

Temperature Over Time:
Temperature Over Time

ARIMA Forecast:
ARIMA Forecast

Anomaly Detection:
Anomaly Detection

Feature Importance:
Feature Importance

Geographical Patterns:
Geographical Patterns

Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, please contact:

M.Mohammed Salman

Email: salman14072004@gmail.com

GitHub: salman712225

