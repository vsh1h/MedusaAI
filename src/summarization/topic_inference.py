#detect topics

import json
import joblib
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "data/processed/lda_model.pkl"
VECTORIZER_PATH = BASE_DIR / "data/processed/tfidf_vectorizer.pkl"
TOPICS_PATH = BASE_DIR / "data/processed/topics.json"

# Load trained artifacts 

print("Loading topic model artifacts...")

lda_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics_data = json.load(f)

print("Artifacts loaded successfully.")
print(f"Model topics: {lda_model.n_components}")
print(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")



# Topic Prediction

def predict_topic(text: str):
    """
    Predict topic for new unseen document

    Steps:
    1) Convert text -> TF-IDF using trained vectorizer
    2) Project into LDA topic space
    3) Pick highest probability topic
    """

    if not text or not text.strip():
        return None, []

    
    vec = vectorizer.transform([text])

    # topic distribution's probabilities
    topic_distribution = lda_model.transform(vec)[0]

    topic_id = int(np.argmax(topic_distribution))

    topic_key = f"topic_{topic_id}"
    keywords = topics_data[topic_key]["words"]

    return topic_id, keywords


def get_topic_distribution(text: str):
    if not text.strip():
        return None

    vec = vectorizer.transform([text])
    return lda_model.transform(vec)[0]


#testing 
if __name__ == "__main__":
    sample_text = """
    We study quantum entanglement in many body systems and analyze
    the phase transition behavior in spin chains using numerical simulations.
    """

    topic_id, words = predict_topic(sample_text)

    print("\nPredicted Topic:", topic_id)
    print("Keywords:", words)