#generate summary
import json
import re
from pathlib import Path

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from topic_inference import predict_topic, get_topic_distribution


# clean text
def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# automatic project path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "arxiv-metadata-oai-snapshot.json"



def stream_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue



def get_text(paper):
    text = paper.get("abstract", "")
    return normalize_whitespace(text)

#sentence split
def split_sentences(text):
    return sent_tokenize(text)


# sentence ranking and topic aware
def rank_sentences(sentences, topic_keywords, topic_distribution):

    if len(sentences) == 0:
        return []

    # sentence TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences).toarray()

    scores = []

    for i, sentence in enumerate(sentences):

        # Information Density
        info_score = np.sum(tfidf_matrix[i])

        # Topic relevance
        words = word_tokenize(sentence.lower())
        topic_overlap = sum(1 for w in words if w in topic_keywords)

        topic_strength = np.max(topic_distribution)
        topic_score = topic_overlap * (1 + topic_strength * 3)

        # position bias 
        if i == 0 or i == len(sentences) - 1:
            position_score = 2
        else:
            position_score = 0

        # Length normalization
        length_score = min(len(words) / 20, 1.5)

        total_score = info_score + topic_score + position_score + length_score
        scores.append(total_score)

    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices


#generate summary
def generate_summary(sentences, topic_keywords, topic_distribution, top_n=3):

    if len(sentences) <= top_n:
        return " ".join(sentences)

    ranked_indices = rank_sentences(sentences, topic_keywords, topic_distribution)

    # take best sentences
    top_indices = ranked_indices[:top_n]

    # keep original order for readability
    top_indices = sorted(top_indices)

    top_sentences = [sentences[i] for i in top_indices]
    return " ".join(top_sentences)


# testing
if __name__ == "__main__":

    docs = stream_documents(DATA_PATH)

    first_paper = next(docs)
    text = get_text(first_paper)

    if not text:
        print("No abstract found.")
        exit()

    sentences = split_sentences(text)

    #Topic detection
    topic_id, keywords = predict_topic(text)
    topic_distribution = get_topic_distribution(text)

    print("\nDetected Topic:", topic_id)
    print("Topic Keywords:", keywords)

    #Generate summary
    summary = generate_summary(sentences, keywords, topic_distribution, top_n=3)

    print("\n===== GENERATED SUMMARY =====\n")
    print(summary)