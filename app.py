from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load ML model
model, vectorizer = pickle.load(open("emotion_model.pkl", "rb"))

def detect_emotion(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data["message"].lower()

    emotion = detect_emotion(user_text)

    if emotion in ["hello", "hy", "hi", "hii", "hiiii", "hiiiiii", "hiiiiiii"]:
        reply = "Hello! How can I help you?"
    elif emotion in ["sad", "I feel very sad", "I am depressed", "I feel lonely", "i am upset"]:
        reply = "I'm sorry you're feeling sad 😔. I'm here."
    elif emotion == "happy":
        reply = "That's great! I'm happy for you 😄"
    elif emotion == "angry":
        reply = "I sense anger. Take a deep breath 🌬️"
    elif emotion == "neutral":
        reply = "okay fine🙂"
    else:
        reply = "I understand. Tell me more 🙂"

    return jsonify({
        "emotion": emotion,
        "reply": reply
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

