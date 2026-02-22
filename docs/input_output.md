## Input

The system accepts the following inputs:

### 1. Research Topic
A user-provided keyword or phrase used to filter or analyze research papers.

Example:
"Graph Neural Networks"

### 2. Research Documents

The documents for analysis can be provided in two ways:

- Uploaded by the user (PDF or TXT)
OR
- Selected from the arXiv dataset

From the dataset, the following fields are used:
- title
- abstract


## Processing Steps

1. Text Preprocessing
   - Tokenization
   - Stop-word removal
   - Lemmatization

2. Feature Extraction
   - TF-IDF / Bag of Words

3. Topic Identification
   - LDA / NMF or K-Means clustering

4. Keyword Extraction
   - Based on TF-IDF scores

5. Extractive Summarization
   - Sentence ranking


## Output

The system produces:

1. Key Terms
2. Identified Topics or Clusters
3. Extractive Summary