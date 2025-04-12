# emotion_bridge.py

from loosh_emotion_model import load_emotion_model, analyze_emotion

def emotion_to_action(emotion):
    """
    Maps emotion to robot behavior.
    Customize this map however you'd like.
    """
    emotion_map = {
        "joy": "DANCE",
        "anger": "TURN_LEFT",
        "sadness": "PAUSE",
        "fear": "RETREAT",
        "neutral": "IDLE",
        "surprise": "MOVE_FORWARD",
        "admiration": "SPIN",
    }

    return emotion_map.get(emotion, "IDLE")

def get_primary_action(text_input):
    classifier = load_emotion_model()
    emotions = analyze_emotion(text_input, classifier, top_n=1)
    top_emotion = emotions[0]['label']
    print(f"Detected Emotion: {top_emotion}")
    return emotion_to_action(top_emotion)

if __name__ == "__main__":
    user_input = input("What are you feeling?\n> ")
    behavior = get_primary_action(user_input)
    print(f"Robot should: {behavior}")
