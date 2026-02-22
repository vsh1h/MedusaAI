import json
from pathlib import Path
from tqdm import tqdm

from summarizer import generate_summary, split_sentences, normalize_whitespace
from topic_inference import predict_topic


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT = BASE_DIR / "data/raw/arxiv-metadata-oai-snapshot.json"
OUTPUT = BASE_DIR / "data/processed/papers_with_summary.jsonl"


def stream_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                continue


def process_paper(paper):
    text = normalize_whitespace(paper.get("abstract", ""))

    # skip only if truly empty
    if len(text.strip()) < 50:
        return None

    sentences = split_sentences(text)

    # fallback if tokenizer fails
    if len(sentences) < 2:
        summary = text[:400]
        topic_id = -1
        keywords = []
    else:
        topic_id, keywords = predict_topic(text)
        summary = generate_summary(sentences, keywords, top_n=3)

    return {
        "id": paper.get("id"),
        "title": paper.get("title"),
        "abstract": text,
        "summary": summary,
        "topic": topic_id,
        "keywords": keywords
    }


def main(limit=5000):
    written = 0

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for i, paper in enumerate(tqdm(stream_documents(INPUT))):

            if i >= limit:
                break

            result = process_paper(paper)

            if result:
                out.write(json.dumps(result) + "\n")
                written += 1

    print("\nDONE")
    print("Total written papers:", written)
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()