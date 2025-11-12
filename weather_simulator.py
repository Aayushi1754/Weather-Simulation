
# weather_simulator_integrated.py

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import newdistribution      # your module for extracting dataset parameters
import markov_model         # module for Markov transition logic
import markov_visualization # module for plotting transition matrix
import visualization        # module for plotting weather results
import evaluation           # module for evaluation functions

def simulate_weather(num_days=30):
    print("\n🌤 Starting Full Weather Simulation...\n")

    # Step 1: Extract Parameters from Dataset
    
    params = newdistribution.get_weather_parameters()
    mean_temp = params["mean_temp"]
    std_temp = params["std_temp"]
    mean_hum = params["mean_humidity"]
    std_hum = params["std_humidity"]
    lambda_rain = params["rain_lambda"]
    p_storm = params["storm_prob"]

    print("📊 Dataset Parameters:")
    print(f"Mean Temp: {mean_temp:.2f}, Std Temp: {std_temp:.2f}")
    print(f"Mean Humidity: {mean_hum:.2f}, Std Humidity: {std_hum:.2f}")
    print(f"λ (Rain Events): {lambda_rain:.2f}, p (Storm): {p_storm:.2f}")

    # Step 2: Build Markov Transition Matrix
    
    transition_matrix = markov_model.build_transition_matrix()
    print("\n✅ Sample Transition Probabilities:")
    for s, probs in list(transition_matrix.items())[:3]:
        print(f"{s}: {probs}")

    # Step 3: Prepare Dates
    
    start_date = newdistribution.get_last_date() + timedelta(days=1)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]

    # Step 4: Simulate Weather
    
    states = list(transition_matrix.keys())
    current_state = random.choice(states)
    np.random.seed(42)
    random.seed(42)

    weather_data = []
    for day, date in enumerate(dates, start=1):
        temp = np.random.normal(mean_temp, std_temp)
        hum = np.random.normal(mean_hum, std_hum)
        rain = np.random.poisson(lambda_rain)
        thunder = np.random.binomial(1, p_storm)

        weather_data.append({
            "Date": date,
            "Day": day,
            "Condition": current_state,
            "Temperature (°C)": round(temp, 2),
            "Humidity (%)": round(hum, 2),
            "Rainfall (events)": rain,
            "Thunderstorm": bool(thunder)
        })

        # Next state using Markov chain
        next_states = list(transition_matrix[current_state].keys())
        probs = list(transition_matrix[current_state].values())
        current_state = random.choices(next_states, weights=probs, k=1)[0]

    
    # Step 5: Convert to DataFrame & Display
    
    sim_df = pd.DataFrame(weather_data)
    print("\n🌦 Simulated Weather Data (First 10 Days):")
    print(sim_df.head(10))

    
    # Step 6: Visualize & Evaluate
    
    evaluation.evaluate_simulation(sim_df)             # prints evaluation metrics
    markov_visualization.plot_transition_matrix(transition_matrix) # plots Markov matrix
    visualization.plot_simulation_results(sim_df)      # plots temperature/humidity

    print("\n🏁 Simulation Complete!\n")


if __name__ == "__main__":
    simulate_weather(30)
