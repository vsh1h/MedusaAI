import argparse
import csv
import json
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_INPUT = Path("data/processed/papers_clean.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")

def _parse_df_value(raw_value):
	value = str(raw_value).strip()
	if value.isdigit():
		return int(value)
	try:
		return float(value)
	except ValueError as err:
		raise argparse.ArgumentTypeError(
			"Value must be an integer count (e.g. 2) or a float proportion (e.g. 0.9)"
		) from err

def _to_text(value):
	if isinstance(value, str):
		return value.strip()
	if isinstance(value, list):
		return " ".join(str(item) for item in value if str(item).strip())
	return ""

def _extract_text(record):
	for key in ("text", "cleaned_text", "processed_text", "abstract"):
		if key in record:
			text = _to_text(record[key])
			if text:
				return text

	token_fields = []
	for key in (
		"tokens",
		"cleaned_tokens",
		"title_tokens",
		"abstract_tokens",
	):
		if key in record and record[key]:
			token_fields.append(_to_text(record[key]))

	return " ".join(part for part in token_fields if part).strip()

def _extract_doc_id(record, fallback_index):
	return str(
		record.get("id")
		or record.get("paper_id")
		or record.get("doc_id")
		or fallback_index
	)

def load_documents(input_path, max_docs=None):
	input_path = Path(input_path)
	documents = []
	doc_ids = []

	if input_path.suffix == ".jsonl":
		with input_path.open("r", encoding="utf-8") as infile:
			for index, line in enumerate(infile, start=1):
				if max_docs and len(documents) >= max_docs:
					break

				if index % 5000 == 0:
					print(f"  Loaded {len(documents)} documents...", flush=True)

				line = line.strip()
				if not line:
					continue

				record = json.loads(line)
				text = _extract_text(record)
				if not text:
					continue

				documents.append(text)
				doc_ids.append(_extract_doc_id(record, index))

	elif input_path.suffix == ".csv":
		with input_path.open("r", encoding="utf-8", newline="") as infile:
			reader = csv.DictReader(infile)
			for index, record in enumerate(reader, start=1):
				if max_docs and len(documents) >= max_docs:
					break

				text = _extract_text(record)
				if not text:
					continue

				documents.append(text)
				doc_ids.append(_extract_doc_id(record, index))

	else:
		raise ValueError("Supported input formats are .jsonl and .csv")

	return documents, doc_ids

def vectorize_documents(documents, max_features=1000, min_df=10, max_df=0.7):
	vectorizer = TfidfVectorizer(
		max_features=max_features,
		min_df=min_df,
		max_df=max_df,
	)
	matrix = vectorizer.fit_transform(documents)
	return vectorizer, matrix

def save_artifacts(matrix, vectorizer, doc_ids, output_dir):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	matrix_path = output_dir / "document_term_matrix_tfidf.npz"
	vocab_path = output_dir / "tfidf_vocabulary.json"
	metadata_path = output_dir / "tfidf_documents.json"

	sparse.save_npz(matrix_path, matrix)

	with vocab_path.open("w", encoding="utf-8") as outfile:
		json.dump(vectorizer.get_feature_names_out().tolist(), outfile)

	with metadata_path.open("w", encoding="utf-8") as outfile:
		json.dump(doc_ids, outfile)

	return matrix_path, vocab_path, metadata_path

def run(
	input_path=DEFAULT_INPUT,
	output_dir=DEFAULT_OUTPUT_DIR,
	max_features=1000,
	max_docs=None,
	min_df=2,
	max_df=0.9,
):
	documents, doc_ids = load_documents(input_path, max_docs=max_docs)
	if not documents:
		raise ValueError("No usable documents were found in the input file")

	vectorizer, matrix = vectorize_documents(
		documents,
		max_features=max_features,
		min_df=min_df,
		max_df=max_df,
	)
	matrix_path, vocab_path, metadata_path = save_artifacts(
		matrix=matrix,
		vectorizer=vectorizer,
		doc_ids=doc_ids,
		output_dir=output_dir,
	)

	print(f"\nLoaded documents: {len(documents)}")
	print(f"Vocabulary size (filtered): {len(vectorizer.get_feature_names_out())}")
	print(f"Document-term matrix shape: {matrix.shape}")
	print(f"Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])):.1f}%")
	print(f"Document importance constraints: min_df={min_df}, max_df={max_df}")
	print(f"\nSaved artifacts:")
	print(f"  Matrix: {matrix_path}")
	print(f"  Vocabulary: {vocab_path}")
	print(f"  Document IDs: {metadata_path}")

def parse_args():
	parser = argparse.ArgumentParser(
		description="Create TF-IDF document-term matrix from cleaned paper data"
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help="Path to cleaned input file (.jsonl or .csv)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory to store TF-IDF artifacts",
	)
	parser.add_argument(
		"--max-features",
		type=int,
		default=1000,
		help="Maximum vocabulary size for TF-IDF",
	)
	parser.add_argument(
		"--max-docs",
		type=int,
		default=None,
		help="Optional cap on number of documents (for quick runs)",
	)
	parser.add_argument(
		"--min-df",
		type=_parse_df_value,
		default=2,
		help="Ignore terms appearing in fewer than this many docs (or proportion)",
	)
	parser.add_argument(
		"--max-df",
		type=_parse_df_value,
		default=0.9,
		help="Ignore terms appearing in more than this doc proportion",
	)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	run(
		input_path=args.input,
		output_dir=args.output_dir,
		max_features=args.max_features,
		max_docs=args.max_docs,
		min_df=args.min_df,
		max_df=args.max_df,
	)
