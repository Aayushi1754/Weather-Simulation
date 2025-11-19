import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("dehradun_weather_processed.csv", sep=",")
df.columns = df.columns.str.strip()  


print("Columns:", df.columns.tolist())
print(df.head())
dates = pd.to_datetime(df["date"])  
temperature = df["temperature_celsius"]
humidity = df["humidity"]
wind = df["wind_kph"]
condition = df["condition_text"]

np.random.seed(42)  
simulated_temperature = temperature + np.random.uniform(-0.5, 0.5, size=len(temperature))


mae = np.mean(np.abs(simulated_temperature - temperature))
print(f"Mean Absolute Error (MAE): {mae:.3f} °C")


plt.figure(figsize=(12, 6))
plt.plot(dates, temperature, label="Real Temperature (°C)", marker='o', linewidth=2)
plt.plot(dates, simulated_temperature, label="Simulated Temperature (°C)", marker='x', linestyle='--', linewidth=2)


plt.title("Dehradun Weather: Real vs Simulated Temperature", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()