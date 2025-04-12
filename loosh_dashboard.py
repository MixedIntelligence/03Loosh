# loosh_dashboard.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_emotions(filename="emotion_log.csv"):
    df = pd.read_csv(filename, names=["timestamp", "text", "emotion", "score"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Aggregate by time and emotion
    pivot = df.pivot_table(values='score', index='timestamp', columns='emotion', aggfunc='mean')

    # Plot
    pivot.plot(figsize=(12, 6), marker='o')
    plt.title("🧠 Loosh Emotional Landscape Over Time")
    plt.xlabel("Time")
    plt.ylabel("Emotion Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_emotions()
