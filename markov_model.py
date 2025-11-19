import pandas as pd
import numpy as np
import random

df = pd.read_csv("dehradun_weather_processed.csv")
condition = df["condition_text"].tolist()

states = list(set(condition))
transition_counts = {s: {s2: 0 for s2 in states} for s in states}

for today, tomorrow in zip(condition[:-1], condition[1:]):
    transition_counts[today][tomorrow] += 1
transition_matrix = {}
for s in states:
    total = sum(transition_counts[s].values())
    transition_matrix[s] = {s2: transition_counts[s][s2]/total if total>0 else 0 for s2 in states}

print("\nSample Markov Chain Transition Matrix (first 5 states):")
for s in list(transition_matrix.keys())[:5]:
    print(f"{s}: {transition_matrix[s]}")
mean_temp = np.mean(df["temperature_celsius"])
std_temp = np.std(df["temperature_celsius"])
mean_humidity = np.mean(df["humidity"])
std_humidity = np.std(df["humidity"])
lam_rain_events = max(0.5, np.mean(df["wind_kph"]))
storm_prob = 0.1

num_days = 30
weather_data = []
current_state = random.choice(states)

for day in range(num_days):
    temp = np.random.normal(mean_temp, std_temp)
    hum = np.random.normal(mean_humidity, std_humidity)
    rain = np.random.poisson(lam_rain_events)
    thunder = np.random.binomial(1, storm_prob)
    
    weather_data.append({
        "day": day+1,
        "condition": current_state,
        "temperature": round(temp,1),
        "humidity": round(hum,1),
        "rain": rain,
        "thunder": bool(thunder)
    })
    
    next_states = list(transition_matrix[current_state].keys())
    probs = list(transition_matrix[current_state].values())
    current_state = random.choices(next_states, weights=probs, k=1)[0]

sim_df = pd.DataFrame(weather_data)
print("\nSimulated Weather Data (first 10 days):")
print(sim_df.head(10))