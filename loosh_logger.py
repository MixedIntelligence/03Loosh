# loosh_logger.py

import csv
from datetime import datetime
from loosh_emotion_model import load_emotion_model, analyze_emotion

LOG_FILE = "emotion_log.csv"

def log_emotion(text, top_n=3):
    classifier = load_emotion_model()
    results = analyze_emotion(text, classifier, top_n=top_n)
    timestamp = datetime.utcnow().isoformat()

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        for entry in results:
            writer.writerow([timestamp, text, entry['label'], f"{entry['score']:.3f}"])
    print("✅ Emotion(s) logged.")

if __name__ == "__main__":
    entry = input("📝 What's on your mind?\n> ")
    log_emotion(entry)
