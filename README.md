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

## 🛠️ Project Setup (Local Development)

This project uses a Python virtual environment to manage dependencies.  
Follow the steps below to set up the project on your local machine.

---

### 📌 Prerequisites

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
## 📂 Clone the Repository
Clone the project repository and navigate into it:

```bash
git clone https://github.com/<your-username>/MedusaAI.git
cd MedusaAI
```

## 🛠️ Create Virtual Environment
Create a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 📦 Install Dependencies
Install the required Python packages:

Upgrade pip:
```bash
pip3 install --upgrade pip
```

```bash
pip3 install -r requirements.txt
```

## 🧠 Download NLP Resources
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

## ▶️ Run a Sample Script
Verify that the setup is working correctly:
```bash
python3 src/sample_run.py
```

Expected output:
All imports working!

## 📦 Dataset Setup

Due to size constraints, the raw dataset is not stored in this repository.

1. Download the dataset from:
   [link here — https://www.kaggle.com/datasets/Cornell-University/arxiv]

2. Place the file in:
   data/raw/

3. Put the data/raw/ folder in .gitignore to avoid committing large files.

To get cleaned data in data/processed/, run the following script:
```bash
python3 src/preprocessing/clean_metadata.py
```