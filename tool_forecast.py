import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(12)

dates=pd.date_range("2022-01-01", periods=48 ,freq="MS")
month=dates.month

trend=np.linspace(300,700,48)
season=np.array([0.9, 0.9, 1.0, 1.1, 1.0, 0.8, 0.6, 0.6, 0.9, 1.5, 1.3, 1.0])
season_effect=season[month-1]

sale_months = ["2022-01", "2022-10", "2023-01", "2023-09", "2023-12",
               "2024-01", "2024-10", "2024-08", "2025-01", "2025-09", "2025-11"]
sale_event = dates.strftime("%Y-%m").isin(sale_months).astype(int)

noise = np.random.normal(0,30,48)
units = trend * season_effect * (1 + 0.35 * sale_event) + noise


df = pd.DataFrame({
    "date": dates,
    "month": month,
    "sale_event": sale_event,
    "units": units.round().astype(int)
})

print(df.head(12))
print()
print(df.groupby("month")["units"].mean().round())
print()
print(df.groupby("sale_event")["units"].mean().round())

# 1. lag features
df["lag_1"] = df["units"].shift(1)
df["lag_12"] = df["units"].shift(12)
df["sale_last_year"] = df["sale_event"].shift(12) 
df = df.dropna()

train = df[df["date"]<"2025-01-01"]
test = df[df["date"] >="2025-01-01"]


# 3. features: NEW → include sale_event
features = ["month", "sale_event","sale_last_year","lag_1","lag_12"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(train[features], train["units"])     
pred = model.predict(test[features])

print("Model MAE:", round(mean_absolute_error(test["units"], pred)))
print("Baseline MAE:", round(mean_absolute_error(test["units"], test["lag_12"])))
    
result = test[["date", "sale_event", "units", "lag_12"]].copy()
result["model_pred"] = pred.round()
print(result)

# --- Version 2: predict the change ---
df["change"] = df["units"] - df["lag_12"]

train = df[df["date"] < "2025-01-01"]
test = df[df["date"] >= "2025-01-01"]
features2 = ["month", "sale_event", "sale_last_year"]
model2 = RandomForestRegressor(n_estimators=200, random_state=42)
model2.fit(train[features2], train["change"])

change_pred = model2.predict(test[features2])
pred2 = test["lag_12"] + change_pred

print("V2 Model MAE:", round(mean_absolute_error(test["units"],pred2)))



