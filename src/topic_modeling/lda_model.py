import argparse
import json
import joblib
from pathlib import Path
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation

DEFAULT_MATRIX_PATH = Path("data/processed/document_term_matrix_tfidf.npz")
DEFAULT_VOCAB_PATH = Path("data/processed/tfidf_vocabulary.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")

def load_tfidf_artifacts(matrix_path, vocab_path):
	"""Load TF-IDF matrix and vocabulary."""
	matrix_path = Path(matrix_path)
	vocab_path = Path(vocab_path)

	if not matrix_path.exists():
		raise FileNotFoundError(f"Matrix not found: {matrix_path}")
	if not vocab_path.exists():
		raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

	matrix = sparse.load_npz(matrix_path)
	with vocab_path.open("r", encoding="utf-8") as f:
		vocabulary = json.load(f)

	return matrix, vocabulary

def fit_lda(matrix, n_topics=5, random_state=42, max_iter=20):
	"""Fit Latent Dirichlet Allocation model."""
	lda = LatentDirichletAllocation(
		n_components=n_topics,
		random_state=random_state,
		max_iter=max_iter,
		learning_method="online",
		n_jobs=-1,
	)
	lda.fit(matrix)
	return lda

def extract_top_words(lda, vocabulary, n_words=10):
	"""Extract top N words for each topic."""
	topics = {}

	for topic_idx, topic_weights in enumerate(lda.components_):
		top_indices = topic_weights.argsort()[-n_words:][::-1]
		top_words = [vocabulary[idx] for idx in top_indices]
		top_scores = [float(topic_weights[idx]) for idx in top_indices]

		topics[f"topic_{topic_idx}"] = {
			"words": top_words,
			"scores": top_scores,
		}

	return topics

def generate_semantic_labels(topics):
	"""Generate human-readable semantic labels from top words."""
	labels = {}
	for topic_id, topic_data in topics.items():
		top_3_words = topic_data["words"][:3]
		label = " + ".join(top_3_words)
		labels[topic_id] = label
	return labels

def save_lda_model(lda, model_path):
	"""Save fitted LDA model to pickle file."""
	model_path = Path(model_path)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(lda, model_path)
	return model_path

def save_topics(topics, output_dir):
	"""Save topics to topics.json."""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	topics_path = output_dir / "topics.json"
	with topics_path.open("w", encoding="utf-8") as f:
		json.dump(topics, f, indent=2)

	return topics_path

def run(
	matrix_path=DEFAULT_MATRIX_PATH,
	vocab_path=DEFAULT_VOCAB_PATH,
	output_dir=DEFAULT_OUTPUT_DIR,
	n_topics=5,
	n_words=10,
	max_iter=20,
):
	"""Run full LDA topic modeling pipeline."""
	print(f"Loading TF-IDF artifacts from {matrix_path}...")
	matrix, vocabulary = load_tfidf_artifacts(matrix_path, vocab_path)
	print(f"  Matrix shape: {matrix.shape}")
	print(f"  Vocabulary size: {len(vocabulary)}")

	print(f"\nFitting LDA with K={n_topics} topics...")
	lda = fit_lda(matrix, n_topics=n_topics, max_iter=max_iter)
	print(f"  LDA model fitted successfully")

	print(f"\nExtracting top {n_words} keywords per topic...")
	topics = extract_top_words(lda, vocabulary, n_words=n_words)
	topics["semantic_labels"] = generate_semantic_labels(topics)

	topics_path = save_topics(topics, output_dir)
	print(f"  Topics saved to: {topics_path}")

	model_path = save_lda_model(lda, output_dir / "lda_model.pkl")
	print(f"  LDA model saved to: {model_path}")

	print(f"\n{'='*60}")
	print("TOPIC MODELING RESULTS")
	print(f"{'='*60}")
	for topic_id, topic_data in topics.items():
		if topic_id == "semantic_labels":
			continue
		words_str = ", ".join(topic_data["words"])
		label = topics["semantic_labels"].get(topic_id, "N/A")
		print(f"\n{topic_id}: '{label}'")
		print(f"  Keywords: {words_str}")
	print(f"{'='*60}\n")

	return topics_path

def parse_args():
	parser = argparse.ArgumentParser(
		description="Run LDA topic modeling on TF-IDF matrix"
	)
	parser.add_argument(
		"--matrix",
		type=Path,
		default=DEFAULT_MATRIX_PATH,
		help="Path to TF-IDF matrix (NPZ file)",
	)
	parser.add_argument(
		"--vocab",
		type=Path,
		default=DEFAULT_VOCAB_PATH,
		help="Path to vocabulary JSON file",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory to save topics.json",
	)
	parser.add_argument(
		"--n-topics",
		type=int,
		default=5,
		help="Number of topics (K parameter)",
	)
	parser.add_argument(
		"--n-words",
		type=int,
		default=10,
		help="Top N words to extract per topic",
	)
	parser.add_argument(
		"--max-iter",
		type=int,
		default=20,
		help="Maximum iterations for LDA training",
	)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	run(
		matrix_path=args.matrix,
		vocab_path=args.vocab,
		output_dir=args.output_dir,
		n_topics=args.n_topics,
		n_words=args.n_words,
		max_iter=args.max_iter,
	)