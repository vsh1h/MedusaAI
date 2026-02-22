
import json
from pathlib import Path
from tqdm import tqdm
from summarizer import get_text, split_sentences, generate_summary
from topic_inference import predict_topic, get_topic_distribution

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data/raw/arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = BASE_DIR / "data/processed/papers_with_summary.jsonl"


# stream dataset safely
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


def process_papers(limit=None):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:

        for i, paper in enumerate(tqdm(stream_documents(INPUT_PATH))):

            if limit and i >= limit:
                break

            # paper id
            paper_id = paper.get("id", f"doc_{i}")

            # extract text
            text = get_text(paper)
            if not text:
                continue

            sentences = split_sentences(text)
            if len(sentences) < 3:
                continue

            try:
                # topic detection
                topic_id, keywords = predict_topic(text)

                # topic Distribution
                topic_distribution = get_topic_distribution(text)

                if topic_distribution is None:
                    continue

                # summary
                summary = generate_summary(
                    sentences,
                    keywords,
                    topic_distribution,
                    top_n=3
                )

                title = paper.get("title", "No Title")  # Extract title from paper
                abstract = paper.get("abstract", "")  # Extract abstract from paper

                record = {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract[:600],   
                    "topic": int(topic_id),
                    "summary": summary
                }

                out.write(json.dumps(record) + "\n")

            except Exception:
                # skip problematic papers
                continue


if __name__ == "__main__":
    # first test on 200 papers
    process_papers(limit=200)
