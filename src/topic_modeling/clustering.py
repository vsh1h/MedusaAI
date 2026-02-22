import argparse
import json
from pathlib import Path
from scipy import sparse
from sklearn.cluster import KMeans
import numpy as np

DEFAULT_MATRIX_PATH = Path("data/processed/document_term_matrix_tfidf.npz")
DEFAULT_DOCS_PATH = Path("data/processed/tfidf_documents.json")
DEFAULT_CORPUS_PATH = Path("data/processed/papers_clean.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed")

def load_tfidf_artifacts(matrix_path, docs_path):
	"""Load TF-IDF matrix and document IDs."""
	matrix_path = Path(matrix_path)
	docs_path = Path(docs_path)

	if not matrix_path.exists():
		raise FileNotFoundError(f"Matrix not found: {matrix_path}")
	if not docs_path.exists():
		raise FileNotFoundError(f"Document IDs not found: {docs_path}")

	matrix = sparse.load_npz(matrix_path)
	with docs_path.open("r", encoding="utf-8") as f:
		doc_ids = json.load(f)

	return matrix, doc_ids

def load_paper_metadata(corpus_path, doc_ids):
	"""Load paper metadata (title, abstract) for enrichment."""
	corpus_path = Path(corpus_path)
	metadata = {}

	if not corpus_path.exists():
		print(f"Warning: Corpus not found at {corpus_path}. Clustering without metadata.")
		return metadata

	print("Loading paper metadata from corpus...")
	with corpus_path.open("r", encoding="utf-8") as f:
		for idx, line in enumerate(f, 1):
			record = json.loads(line)
			doc_id = record.get("id")
			if doc_id and doc_id in doc_ids:
				# Smart title fallback: title > text (truncated) > "Untitled Paper"
				title = record.get("title")
				if not title:
					text = record.get("text", "")
					title = text[:100] + "..." if text else "Untitled Paper"
				
				metadata[doc_id] = {
					"title": title,
					"authors": record.get("authors", "Unknown"),
					"year": record.get("year", "Unknown"),
					"categories": record.get("categories", []),
				}

			if idx % 20000 == 0:
				print(f"  Processed {idx} records...")

	print(f"Loaded metadata for {len(metadata)} papers")
	return metadata

def run_kmeans(matrix, n_clusters=5, random_state=42, n_init=10):
	"""Run K-Means clustering on TF-IDF matrix."""
	print(f"Running K-Means with K={n_clusters}...")
	kmeans = KMeans(
		n_clusters=n_clusters,
		random_state=random_state,
		n_init=n_init,
		max_iter=300,
	)
	cluster_labels = kmeans.fit_predict(matrix)
	return kmeans, cluster_labels

def create_paper_clusters(doc_ids, cluster_labels, metadata, kmeans):
	"""Create labeled cluster assignments for each paper."""
	papers_by_cluster = {f"cluster_{i}": [] for i in range(len(np.unique(cluster_labels)))}

	for doc_id, cluster_label in zip(doc_ids, cluster_labels):
		paper_info = {
			"id": doc_id,
			"cluster": int(cluster_label),
		}

		# Add metadata if available
		if doc_id in metadata:
			paper_info.update(metadata[doc_id])

		cluster_key = f"cluster_{cluster_label}"
		papers_by_cluster[cluster_key].append(paper_info)

	return papers_by_cluster

def compute_cluster_stats(papers_by_cluster, matrix, cluster_labels):
	"""Compute statistics for each cluster."""
	stats = {}
	for cluster_id in range(len(papers_by_cluster)):
		cluster_mask = cluster_labels == cluster_id
		cluster_size = np.sum(cluster_mask)
		cluster_density = sparse.csr_matrix(matrix[cluster_mask]).mean()

		stats[f"cluster_{cluster_id}"] = {
			"size": int(cluster_size),
			"density": float(cluster_density),
			"avg_documents": int(cluster_size),
		}

	return stats

def save_clustering_results(papers_by_cluster, stats, output_dir):
	"""Save cluster assignments and statistics."""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Save labeled papers
	papers_path = output_dir / "papers_clusters.json"
	with papers_path.open("w", encoding="utf-8") as f:
		json.dump(papers_by_cluster, f, indent=2)

	# Save cluster statistics
	stats_path = output_dir / "cluster_stats.json"
	with stats_path.open("w", encoding="utf-8") as f:
		json.dump(stats, f, indent=2)

	return papers_path, stats_path

def run(
	matrix_path=DEFAULT_MATRIX_PATH,
	docs_path=DEFAULT_DOCS_PATH,
	corpus_path=DEFAULT_CORPUS_PATH,
	output_dir=DEFAULT_OUTPUT_DIR,
	n_clusters=5,
	random_state=42,
):
	"""Run full K-Means clustering pipeline."""
	print("Loading TF-IDF matrix and document IDs...")
	matrix, doc_ids = load_tfidf_artifacts(matrix_path, docs_path)
	print(f"  Matrix shape: {matrix.shape}")
	print(f"  Documents: {len(doc_ids)}\n")

	print("Loading paper metadata...")
	metadata = load_paper_metadata(corpus_path, doc_ids)
	print()

	print("Running K-Means clustering...")
	kmeans, cluster_labels = run_kmeans(matrix, n_clusters=n_clusters, random_state=random_state)
	print(f"  Clustering complete\n")

	print("Creating paper cluster assignments...")
	papers_by_cluster = create_paper_clusters(doc_ids, cluster_labels, metadata, kmeans)
	print(f"  Assigned {len(doc_ids)} papers to {n_clusters} clusters\n")

	print("Computing cluster statistics...")
	stats = compute_cluster_stats(papers_by_cluster, matrix, cluster_labels)

	papers_path, stats_path = save_clustering_results(papers_by_cluster, stats, output_dir)

	print(f"\n{'='*60}")
	print("K-MEANS CLUSTERING RESULTS")
	print(f"{'='*60}")
	print(f"Total Papers: {len(doc_ids)}")
	print(f"Number of Clusters: {n_clusters}\n")

	for cluster_id in range(n_clusters):
		cluster_key = f"cluster_{cluster_id}"
		cluster_size = stats[cluster_key]["size"]
		cluster_density = stats[cluster_key]["density"]
		print(f"Cluster {cluster_id}: {cluster_size} papers (density: {cluster_density:.4f})")

	print(f"{'='*60}\n")
	print(f"Saved papers with cluster labels: {papers_path}")
	print(f"Saved cluster statistics: {stats_path}")

	return papers_path, stats_path

def parse_args():
	parser = argparse.ArgumentParser(
		description="Run K-Means clustering on TF-IDF document matrix"
	)
	parser.add_argument(
		"--matrix",
		type=Path,
		default=DEFAULT_MATRIX_PATH,
		help="Path to TF-IDF matrix (NPZ file)",
	)
	parser.add_argument(
		"--docs",
		type=Path,
		default=DEFAULT_DOCS_PATH,
		help="Path to document IDs JSON file",
	)
	parser.add_argument(
		"--corpus",
		type=Path,
		default=DEFAULT_CORPUS_PATH,
		help="Path to papers_clean.jsonl corpus for metadata",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory to save clustering results",
	)
	parser.add_argument(
		"--n-clusters",
		type=int,
		default=5,
		help="Number of clusters (K parameter)",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed for reproducibility",
	)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	run(
		matrix_path=args.matrix,
		docs_path=args.docs,
		corpus_path=args.corpus,
		output_dir=args.output_dir,
		n_clusters=args.n_clusters,
		random_state=args.random_state,
	)
