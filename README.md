# Bus Route Travel Time Prediction

A machine learning project that predicts bus travel time based on real-world factors like traffic, weather, road type, and driver experience.

## What this project does

Given a set of route and environmental conditions, the models predict how long a bus trip will take in minutes. This kind of prediction is useful for school bus scheduling, fleet management, and reducing late arrivals.

## Dataset

2000 records of school bus routes across different zones in the Klang Valley area. Each record includes details about the route, driver, bus condition, weather, and traffic at the time of the trip.

Features used:
- Route info: number of stops, total distance, zone, road type, road quality
- Time info: hour, day of week, season, peak hour flag
- Conditions: traffic level, weather, temperature
- Driver: experience (years), age
- Bus: age, capacity, maintenance score, occupancy rate

Target variable: `total_time_minutes`

## Models

Two models were trained and compared:

| Model | Description |
|-------|-------------|
| XGBoost | Gradient boosted trees, handles tabular data well |
| Neural Network | Built with Keras, 3 dense layers with dropout |

Both models were evaluated using MAE, RMSE, and R² score.

## Project Structure

```
bus-route-travel-time/
├── data/
│   └── bus_route_travel_time.csv
├── outputs/
│   ├── 01_eda.png
│   └── model_comparison.csv
├── models/
│   ├── xgboost_model.json
│   └── neural_network_model.h5
├── notebook.ipynb
└── README.md
```

## How to run

1. Clone the repo
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
   ```
3. Open `notebook.ipynb` and run all cells

## Tech stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras
