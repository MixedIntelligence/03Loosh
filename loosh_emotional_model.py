import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained transformer for emotion analysis
# This one uses a DistilBERT fine-tuned on GoEmotions (27-class emotion dataset)
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return classifier

def analyze_emotion(text, classifier, top_n=3):
    scores = classifier(text)[0]
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    return sorted_scores[:top_n]

if __name__ == "__main__":
    classifier = load_emotion_model()
    
    # Example input – could be from journal, voice-to-text, or chatbot
    sample_text = "I feel disconnected lately, like I’m just floating through tasks. Nothing feels real."

    print(f"📘 Input: {sample_text}\n")
    result = analyze_emotion(sample_text, classifier)

    print("🔎 Detected Emotional States:")
    for entry in result:
        print(f" - {entry['label']}: {entry['score']:.3f}")
