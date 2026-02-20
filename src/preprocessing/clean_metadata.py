import json
from pathlib import Path
from src.ingestion.stream_reader import stream_arxiv_json
import spacy

RAW_PATH = Path("data/raw/arxiv-metadata-oai-snapshot.json")
OUT_PATH = Path("data/processed/papers_clean.jsonl")

BATCH_SIZE = 1000

nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser", "lemmatizer"]
)

def clean_texts(texts):
    """Process a batch of texts with spaCy."""
    for doc in nlp.pipe(texts, batch_size=50):
        tokens = [
            token.text.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        yield " ".join(tokens)

def run():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    buffer_records = []
    buffer_texts = []
    count = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for record in stream_arxiv_json(RAW_PATH):
            title = record.get("title", "").replace("\n", " ").strip()
            abstract = record.get("abstract", "").replace("\n", " ").strip()

            full_text = f"{title} {abstract}"

            buffer_records.append(record)
            buffer_texts.append(full_text)

            if len(buffer_texts) >= BATCH_SIZE:
                for rec, processed in zip(
                    buffer_records,
                    clean_texts(buffer_texts)
                ):
                    cleaned = {
                        "id": rec.get("id"),
                        "text": processed,
                        "categories": rec.get("categories", "").split(),
                        "authors": rec.get("authors", ""),
                        "year": rec.get("update_date", "")[:4]
                    }
                    out.write(json.dumps(cleaned) + "\n")
                    count += 1

                buffer_records.clear()
                buffer_texts.clear()

                if count % 50_000 == 0:
                    print(f"Processed {count} records")

        if buffer_texts:
            for rec, processed in zip(
                buffer_records,
                clean_texts(buffer_texts)
            ):
                cleaned = {
                    "id": rec.get("id"),
                    "text": processed,
                    "categories": rec.get("categories", "").split(),
                    "authors": rec.get("authors", ""),
                    "year": rec.get("update_date", "")[:4]
                }
                out.write(json.dumps(cleaned) + "\n")
                count += 1

    print(f"Finished. Total records: {count}")

if __name__ == "__main__":
    run()