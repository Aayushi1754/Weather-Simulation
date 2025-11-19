import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def simulate_weather(num_days=30):
    # ----- Simulation Parameters -----
    mean_temp = 20.3
    std_temp = 0.63
    mean_humidity = 55.24
    std_humidity = 5.18
    rain_lambda = 5.36
    storm_prob = 0.10

    # ----- Define Markov Chain for weather states -----
    transition_matrix = {
        "Clear": {"Clear": 0.82, "Partially cloudy": 0.18},
        "Partially cloudy": {"Clear": 0.67, "Partially cloudy": 0.33},
    }

    states = list(transition_matrix.keys())
    current_state = np.random.choice(states)

    # ----- Start Date -----
    start_date = datetime(2025, 11, 9)

    # ----- Simulation -----
    data = []
    for day in range(1, num_days + 1):
        # Generate date
        date = start_date + timedelta(days=day - 1)

        # Generate weather values
        temp = np.random.normal(mean_temp, std_temp)
        humidity = np.random.normal(mean_humidity, std_humidity)
        rain = np.random.poisson(rain_lambda)
        thunder = np.random.rand() < storm_prob

        # Save the record
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Day": day,
            "Condition": current_state,
            "Temperature (°C)": round(temp, 2),
            "Humidity (%)": round(humidity, 2),
            "Rainfall (events)": rain,
            "Thunderstorm": thunder
        })

        # Determine next state using Markov transition
        next_state = np.random.choice(
            states,
            p=[
                transition_matrix[current_state]["Clear"],
                transition_matrix[current_state]["Partially cloudy"]
            ]
        )
        current_state = next_state

    # ----- Save to CSV -----
    df = pd.DataFrame(data)
    df.to_csv("simulated_weather.csv", index=False)

    # ----- Display Summary -----
    print("\nSample Markov Chain Transition Matrix:")
    for k, v in transition_matrix.items():
        print(f"{k}: {v}")

    print("\n🌦 Simulated Weather Data (first 10 days):")
    print(df.head(10))
    print("\n✅ Data saved to 'simulated_weather.csv'")
