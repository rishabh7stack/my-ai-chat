from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Training data
texts = [
    "hello","hy","hi","hii","hiiii","hiiiiii","hiiiiiii",
    "I feel very sad", "I am depressed", "I feel lonely",
    "I am very happy", "Today is amazing", "I am excited",
    "I am angry", "I hate this", "This is annoying",
    "Okay", "I am fine", "Nothing much","i dont know what i feel right now",
]

labels = [
    "hello","hy","hi","hii","hiiii","hiiiiii","hiiiiiii",
    "sad", "sad", "sad",
    "happy", "happy", "happy",
    "angry", "angry", "angry",
    "neutral", "neutral", "neutral"
]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save model
pickle.dump((model, vectorizer), open("emotion_model.pkl", "wb"))

print("✅ Model trained and saved as emotion_model.pkl")
