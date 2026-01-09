from flask import Flask, request, jsonify, send_from_directory
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

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data["message"].lower()
    emotion = detect_emotion(user_text)

    if emotion == "sad":
        reply = "I'm sorry you're feeling sad 😔. I'm here."
    elif emotion == "happy":
        reply = "That's great! I'm happy for you 😄"
    elif emotion == "angry":
        reply = "I sense anger. Take a deep breath 🌬️"
    else:
        reply = "I understand. Tell me more 🙂"

    return jsonify({
        "emotion": emotion,
        "reply": reply
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
