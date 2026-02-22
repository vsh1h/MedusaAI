import argparse
import json
from pathlib import Path
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

DEFAULT_CORPUS_PATH = Path("data/processed/papers_clean.jsonl")
DEFAULT_TOPICS_PATH = Path("data/processed/topics.json")
DEFAULT_VOCAB_PATH = Path("data/processed/tfidf_vocabulary.json")
DEFAULT_OUTPUT_DIR = Path("data/processed")

def load_evaluation_data(topics_path, corpus_path, vocab_path=None, sample_size=50000):
	"""Load topics and a sampled corpus for efficient coherence evaluation."""
	with open(topics_path, "r", encoding="utf-8") as f:
		topics = json.load(f)

	# Load cleaned text for Gensim context
	texts = []
	print(f"Loading corpus from papers_clean.jsonl (sample: {sample_size} docs)...")
	with open(corpus_path, "r", encoding="utf-8") as f:
		for idx, line in enumerate(f, 1):
			if idx > sample_size:
				break
			record = json.loads(line)
			text = record.get("text") or record.get("cleaned_text")
			if text:
				# Tokenize by splitting on whitespace
				tokens = text.split()
				texts.append(tokens)
			
			if idx % 10000 == 0:
				print(f"  Loaded {idx} documents...")
	
	print(f"Total documents loaded: {len(texts)}")
	return topics, texts

def calculate_coherence_score(topics, texts, coherence_measure="c_v"):
	"""
	Calculate coherence score using the corpus (texts) for context.
	c_v is generally more stable for academic reports.
	"""
	topic_words_list = []
	for topic_id in sorted(topics.keys()):
		# Skip non-topic entries like "semantic_labels"
		if topic_id == "semantic_labels" or not isinstance(topics[topic_id], dict) or "words" not in topics[topic_id]:
			continue
		topic_data = topics[topic_id]
		words = topic_data["words"]
		topic_words_list.append(words)

	# Create the Gensim dictionary required by the model
	print("Creating Gensim dictionary from corpus...")
	dictionary = Dictionary(texts)
	print(f"Dictionary contains {len(dictionary)} unique tokens")

	try:
		print(f"Calculating {coherence_measure} coherence score...")
		coherence_model = CoherenceModel(
			topics=topic_words_list,
			texts=texts,
			dictionary=dictionary,
			coherence=coherence_measure,
		)
		score = coherence_model.get_coherence()
	except Exception as e:
		raise RuntimeError(f"Failed to calculate {coherence_measure} coherence: {e}") from e

	return score, topic_words_list

def interpret_coherence(score, threshold=0.5):
	"""Provide interpretability guidance based on coherence score."""
	if score > threshold:
		interpretation = "GOOD - Topics appear to capture meaningful structures"
		recommendation = "Topics are interpreable. Consider increasing K for finer-grained analysis."
	elif score > 0.35:
		interpretation = "ACCEPTABLE - Topics have moderate coherence"
		recommendation = "Topics are somewhat coherent. Refine preprocessing or test different K values."
	else:
		interpretation = "POOR - Topics may lack semantic meaning"
		recommendation = "Try decreasing K or adjusting min_df/max_df document frequency filters."

	return interpretation, recommendation

def save_coherence_results(score, interpretation, recommendation, topic_words_list, output_dir, coherence_measure="c_v"):
	"""Save coherence evaluation results to JSON."""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	results = {
		"coherence_score": float(score),
		"coherence_measure": coherence_measure,
		"interpretation": interpretation,
		"recommendation": recommendation,
		"topic_counts": len(topic_words_list),
		"details": {
			"description": f"{coherence_measure} coherence measures how often topic words appear together in the corpus",
			"range": [0.0, 1.0],
			"good_threshold": 0.5,
			"note": "Score is based on actual co-occurrence in papers_clean.jsonl corpus",
		},
	}

	results_path = output_dir / "coherence_results.json"
	with results_path.open("w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	return results_path

def run(
	topics_path=DEFAULT_TOPICS_PATH,
	corpus_path=DEFAULT_CORPUS_PATH,
	vocab_path=DEFAULT_VOCAB_PATH,
	output_dir=DEFAULT_OUTPUT_DIR,
	coherence_measure="c_v",
	threshold=0.5,
	sample_size=50000,
):
	"""Run full coherence evaluation pipeline."""
	print(f"Loading evaluation data...")
	topics, texts = load_evaluation_data(topics_path, corpus_path, vocab_path, sample_size=sample_size)
	print(f"Loaded {len(topics)} topics and {len(texts)} documents\n")

	score, topic_words_list = calculate_coherence_score(
		topics, texts, coherence_measure=coherence_measure
	)

	interpretation, recommendation = interpret_coherence(score, threshold=threshold)
	results_path = save_coherence_results(
		score, interpretation, recommendation, topic_words_list, output_dir, coherence_measure
	)

	print(f"\n{'='*60}")
	print("TOPIC COHERENCE EVALUATION")
	print(f"{'='*60}")
	print(f"Coherence Score ({coherence_measure}): {score:.4f}")
	print(f"Interpretation: {interpretation}")
	print(f"\nRecommendation:")
	print(f"  {recommendation}")
	print(f"{'='*60}\n")
	print(f"Results saved to: {results_path}")

	return results_path

def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate LDA topic coherence using Gensim with corpus context"
	)
	parser.add_argument(
		"--topics",
		type=Path,
		default=DEFAULT_TOPICS_PATH,
		help="Path to topics.json file",
	)
	parser.add_argument(
		"--corpus",
		type=Path,
		default=DEFAULT_CORPUS_PATH,
		help="Path to papers_clean.jsonl corpus file",
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
		help="Directory to save coherence_results.json",
	)
	parser.add_argument(
		"--coherence-measure",
		type=str,
		default="c_v",
		choices=["c_v", "u_mass", "c_uci", "c_npmi"],
		help="Coherence metric to use (c_v recommended for reports)",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.5,
		help="Threshold score for 'good' coherence interpretation",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=50000,
		help="Number of documents to sample from corpus for coherence calculation",
	)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	run(
		topics_path=args.topics,
		corpus_path=args.corpus,
		vocab_path=args.vocab,
		output_dir=args.output_dir,
		coherence_measure=args.coherence_measure,
		threshold=args.threshold,
		sample_size=args.sample_size,
	)
