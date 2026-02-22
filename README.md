# Intelligent Research Topic Analysis System

## Project Overview
This project focuses on building an **AI-based research topic analysis system** using **traditional Natural Language Processing (NLP) and Machine Learning techniques**.  
The system processes research papers related to a user-specified topic, identifies key themes and keywords, applies topic modeling or clustering, and generates **extractive summaries**.

The application includes a **user-friendly interface** and is designed using **only open-source libraries and free tools**, without the use of large language models (LLMs).

---

##  Project Objective
The objective of this project is to:
- Analyze research documents related to a given topic
- Extract important keywords and themes
- Discover latent topics or clusters within documents
- Generate structured **extractive summaries**
- Present analytical insights through a simple UI

This system demonstrates the strengths and limitations of **traditional NLP-based research analysis**.

---

##  Dataset
- **arXiv Research Papers Dataset**
- Source: https://www.kaggle.com/datasets/Cornell-University/arxiv

The dataset contains metadata and abstracts of academic research papers across multiple domains and is used for topic analysis and summarization.

---


##  Workflow (Traditional NLP Pipeline)

Research Papers / Uploaded Documents

↓

Text Preprocessing
(Tokenization, Stopword Removal, Lemmatization)

↓


Feature Extraction
(TF-IDF / Bag of Words)

↓

Topic Modeling / Clustering
(LDA / NMF / KMeans)

↓

Extractive Summarization
(TF-IDF-based Sentence Scoring)

↓

UI Display
(Keywords, Topics, Summary)



---

##  Tech Stack

### Programming Language
- Python

### NLP & Machine Learning
- NLTK
- spaCy
- scikit-learn
- NumPy

### Feature Extraction & Modeling
- TF-IDF
- Bag of Words
- LDA / NMF
- KMeans Clustering

### User Interface
- Streamlit / Gradio

### Visualization 
- matplotlib
- seaborn

## Project Setup (Local Development)

This project uses a Python virtual environment to manage dependencies.  
Follow the steps below to set up the project on your local machine.

---

### Prerequisites

Make sure you have the following installed:
- Python 3.9 or higher
- pip3
- Git
- VS Code (recommended)

Check versions:
```bash
python3 --version
pip3 --version
git --version
```
## Clone the Repository
Clone the project repository and navigate into it:

```bash
git clone https://github.com/<your-username>/MedusaAI.git
cd MedusaAI
```

## Create Virtual Environment
Create a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies
Install the required Python packages:

Upgrade pip:
```bash
pip3 install --upgrade pip
```

```bash
pip3 install -r requirements.txt
```

## Download NLP Resources
Download spaCy's English language model:
```bash
python3 -m spacy download en_core_web_sm
```

NLTK Datasets:

Open the Python shell:
```bash
python3
```
Then run:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
exit()
```

## Run a Sample Script
Verify that the setup is working correctly:
```bash
python3 src/sample_run.py
```

Expected output:
All imports working!

## Dataset Setup

Due to size constraints, the raw dataset is not stored in this repository.

1. Download the dataset from the link provided in the project objective.

2. Place the file in:
   data/raw/

3. Put the data/raw/ folder in .gitignore to avoid committing large files.

4. To get cleaned data in data/processed/, run the following script:
```bash
python3 src/preprocessing/clean_metadata.py
```

Make sure the data/processed/ folder is in .gitignore to avoid committing cleaned data.

---

## Topic Modeling Pipeline (Milestone 1)

This section covers the complete NLP pipeline for vectorization, topic modeling, coherence evaluation, and clustering.

### Prerequisites
Ensure you have completed the dataset setup and have `data/processed/papers_clean.jsonl` available.

---

### Step 1: Vectorization (TF-IDF Feature Extraction)

Convert cleaned text tokens into numerical TF-IDF features with noise filtering.

**Command:**
```bash
python3 -m src.topic_modeling.vectorize --max-docs 100000 --min-df 20 --max-df 0.45 --max-features 1000
```

**Parameters:**
- `--max-docs`: Maximum documents to process (default: 100,000)
- `--min-df`: Minimum document frequency (removes rare words)
- `--max-df`: Maximum document frequency (removes overly common words)
- `--max-features`: Top N features to keep (default: 1000)

**Outputs:**
- `data/processed/document_term_matrix_tfidf.npz` - Sparse TF-IDF matrix
- `data/processed/tfidf_vocabulary.json` - Vocabulary list
- `data/processed/tfidf_documents.json` - Document IDs

**Quick Test (5k docs):**
```bash
python3 -m src.topic_modeling.vectorize --max-docs 5000 --min-df 10 --max-df 0.7
```

---

### Step 2: LDA Topic Modeling

Extract latent topics using Latent Dirichlet Allocation with semantic labeling.

**Command:**
```bash
python3 -m src.topic_modeling.lda_model --n-topics 5 --n-words 10 --max-iter 20
```

**Parameters:**
- `--n-topics`: Number of topics to extract (K parameter, default: 5)
- `--n-words`: Top N keywords per topic (default: 10)
- `--max-iter`: Maximum LDA iterations (default: 20)

**Outputs:**
- `data/processed/topics.json` - Topics with keywords, scores, and semantic labels
- `data/processed/lda_model.pkl` - Serialized LDA model for reuse

**Example Output:**
```
topic_0: 'model + gauge + theory'
  Keywords: model, gauge, theory, energy, quark, mass, matter, neutrino, qcd, gravity
```

---

### Step 3: Coherence Evaluation

Evaluate topic quality using corpus-based coherence metrics.

**Command:**
```bash
python3 -m src.topic_modeling.coherence --sample-size 100000
```

**Parameters:**
- `--sample-size`: Number of documents to sample for coherence calculation (default: 50,000)
- `--coherence-measure`: Metric to use (`c_v`, `u_mass`, `c_uci`, `c_npmi`; default: `c_v`)
- `--threshold`: Score threshold for "good" coherence (default: 0.5)

**Outputs:**
- `data/processed/coherence_results.json` - Coherence score with interpretation

**Interpretation Guide:**
- **> 0.5**: GOOD - Topics capture meaningful structures
- **0.35 - 0.5**: ACCEPTABLE - Moderate coherence
- **< 0.35**: POOR - Topics may lack semantic meaning

---

### Step 4: K-Means Clustering

Group papers into clusters and label them with metadata for dashboard visualization.

**Command:**
```bash
python3 -m src.topic_modeling.clustering --n-clusters 5
```

**Parameters:**
- `--n-clusters`: Number of clusters (default: 5)
- `--random-state`: Random seed for reproducibility (default: 42)

**Outputs:**
- `data/processed/papers_clusters.json` - Papers with cluster assignments and metadata
- `data/processed/cluster_stats.json` - Cluster size and density statistics

**Features:**
- Smart title fallback: `title` → truncated `text` → "Untitled Paper"
- Metadata enrichment: title, authors, year, categories

---

### Complete Pipeline Execution

Run all 4 steps sequentially for 100k documents:

```bash
# Step 1: Vectorization
python3 -m src.topic_modeling.vectorize --max-docs 100000 --min-df 20 --max-df 0.45 --max-features 1000

# Step 2: Topic Modeling
python3 -m src.topic_modeling.lda_model --n-topics 5 --n-words 10 --max-iter 20

# Step 3: Coherence Evaluation
python3 -m src.topic_modeling.coherence --sample-size 100000

# Step 4: Clustering
python3 -m src.topic_modeling.clustering --n-clusters 5
```

**Expected Runtime:**
- Vectorization (100k docs): ~2-3 minutes
- LDA (100k docs): ~5-10 minutes
- Coherence (100k sample): ~3-5 minutes
- Clustering (100k docs): ~1-2 minutes

---

### Verify Output Files

Check that all standard files are generated:

```bash
ls -lh data/processed/{topics.json,lda_model.pkl,coherence_results.json,papers_clusters.json,cluster_stats.json}
```

**Expected Files:**
- `topics.json` - Topic keywords with semantic labels
- `lda_model.pkl` - Trained LDA model
- `coherence_results.json` - Topic quality score
- `papers_clusters.json` - Labeled papers (1.6MB+ for 100k docs)
- `cluster_stats.json` - Cluster distribution

---

### Troubleshooting

**Issue:** `KeyError: 'words'` in coherence.py
- **Fix:** Update to latest version that skips `semantic_labels` key

**Issue:** LDA takes too long on large corpus
- **Fix:** Reduce `--max-docs` for faster iteration (e.g., 5000 for testing)

**Issue:** Memory errors on large datasets
- **Fix:** Use `--sample-size` parameter in coherence step to limit memory usage

---
