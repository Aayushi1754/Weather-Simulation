"""
weather_simulator.py
Integrated Weather Simulation System

Combines:
- evaluation.py (error analysis)
- markov_model.py (state transitions)
- markov_visualization.py (distributions)
- newdistribution.py (Markov + probability-based simulation)
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime, timedelta
from scipy.stats import poisson, binom, norm

# LOAD AND PREPARE DATA

def load_weather_data(filename="dehradun_weather_processed.csv"):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    print(f"Loaded dataset with {len(df)} records and {len(df.columns)} columns.")
    return df

# BUILD MARKOV CHAIN TRANSITION MATRIX

def build_markov_chain(df):
    condition = df["condition_text"].tolist()
    states = list(set(condition))
    transition_counts = {s: {s2: 0 for s2 in states} for s in states}

    for today, tomorrow in zip(condition[:-1], condition[1:]):
        transition_counts[today][tomorrow] += 1

    transition_matrix = {}
    for s in states:
        total = sum(transition_counts[s].values())
        transition_matrix[s] = {s2: transition_counts[s][s2]/total if total > 0 else 1/len(states) for s2 in states}

    print("\nSample Markov Transition Matrix (first 5 states):")
    for s in list(transition_matrix.keys())[:5]:
        print(f"{s}: {transition_matrix[s]}")

    return transition_matrix

# WEATHER SIMULATION (Markov + Statistical Distributions)

def simulate_weather(df, transition_matrix, num_days=30, start_date="2025-11-09"):
    mean_temp = df["temperature_celsius"].mean()
    std_temp = df["temperature_celsius"].std()
    mean_humidity = df["humidity"].mean()
    std_humidity = df["humidity"].std()
    lam_rain_events = max(0.5, df["wind_kph"].mean())
    storm_prob = (df["condition_text"].str.contains("Storm|Thunder", case=False)).mean() or 0.1

    states = list(transition_matrix.keys())
    current_state = random.choice(states)

    data = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    for day in range(num_days):
        date = start + timedelta(days=day)
        temp = np.random.normal(mean_temp, std_temp)
        hum = np.random.normal(mean_humidity, std_humidity)
        rain = np.random.poisson(lam_rain_events)
        thunder = np.random.binomial(1, storm_prob)

        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Day": day + 1,
            "Condition": current_state,
            "Temperature (°C)": round(temp, 2),
            "Humidity (%)": round(hum, 2),
            "Rainfall (events)": rain,
            "Thunderstorm": bool(thunder)
        })

        next_states = list(transition_matrix[current_state].keys())
        probs = list(transition_matrix[current_state].values())
        current_state = random.choices(next_states, weights=probs, k=1)[0]

    sim_df = pd.DataFrame(data)
    sim_df.to_csv("simulated_weather.csv", index=False)

    print("\n✅ Weather simulation completed and saved as 'simulated_weather.csv'.")
    print(sim_df.head(10))
    return sim_df

# VISUALIZATION UTILITIES

def plot_markov_chain(transition_matrix, sim_df=None):
    tm_df = pd.DataFrame(transition_matrix).T.fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    # (a) Transition Graph
    G = nx.DiGraph()
    for s in transition_matrix:
        for s2, p in transition_matrix[s].items():
            if p > 0:
                G.add_edge(s, s2, weight=p)

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000,
            arrows=True, font_size=9, ax=axes[0])
    axes[0].set_title("Markov Chain Transition Graph")

    # (b) Transition Matrix Heatmap
    sns.heatmap(tm_df, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Transition Probability Matrix")

    # (c) Condition Timeline
    if sim_df is not None:
        axes[2].plot(sim_df["Day"], sim_df["Condition"], marker='o', color='teal')
        axes[2].set_title("Simulated Weather Sequence")
        axes[2].set_xlabel("Day")
        axes[2].set_ylabel("Condition")

    # (d) Stationary Distribution
    try:
        P = tm_df.values
        eigvals, eigvecs = np.linalg.eig(P.T)
        stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
        stationary = stationary[:, 0] / stationary[:, 0].sum()
        pd.Series(stationary, index=tm_df.index).plot(kind='bar', ax=axes[3], color='lightgreen')
        axes[3].set_title("Stationary Distribution")
        axes[3].set_ylabel("Probability")
    except Exception as e:
        axes[3].text(0.5, 0.5, f"Error: {e}", ha='center')

    plt.tight_layout()
    plt.show()


def compare_real_vs_simulated(df, sim_df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["temperature_celsius"][:50].reset_index(drop=True), label="Real Temp", marker='o')
    plt.plot(sim_df["Temperature (°C)"][:50].reset_index(drop=True), label="Simulated Temp", marker='x')
    plt.title("Real vs Simulated Temperature (First 50 Days)")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# MAIN EXECUTION

if __name__ == "__main__":
    df = load_weather_data()
    transition_matrix = build_markov_chain(df)
    sim_df = simulate_weather(df, transition_matrix, num_days=30)
    plot_markov_chain(transition_matrix, sim_df)
    compare_real_vs_simulated(df, sim_df)
