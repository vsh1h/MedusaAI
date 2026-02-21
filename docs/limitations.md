# Limitations of the Traditional NLP-Based Research Topic Analysis System

Although the proposed system can extract keywords, identify topics, and generate summaries from research papers, it has several limitations due to the use of classical NLP techniques and the constraints of the dataset.

---

## 1. Lack of Deep Semantic Understanding

Traditional NLP techniques such as TF-IDF and clustering rely on statistical patterns rather than true language understanding.  
The system cannot interpret:

- the actual meaning of the text
- relationships between concepts
- implicit context

As a result, the extracted topics are based on word frequency rather than semantic relevance.

---

## 2. No Reasoning or Analytical Capability

The system cannot:

- compare multiple research papers
- identify research gaps
- draw conclusions
- answer conceptual research questions

It only performs surface-level text analysis.

---

## 3. Extractive Summarization Only

The summarization method selects important sentences directly from the original text.

This means:

- No paraphrasing
- No abstraction
- No generation of new explanations

If the original abstract is poorly written, the summary quality will also be poor.

---

## 4. Static and Dataset-Dependent Knowledge

The system works only on the provided dataset and cannot:

- retrieve recent research papers
- access real-time information
- update its knowledge dynamically

---

## 5. Manual Interpretation of Topics

Topic modeling techniques (such as LDA or clustering) output groups of words.

The system does not automatically generate human-readable labels for these topics.  
Users must interpret the meaning of each topic manually.

---

## 6. Limited Query Understanding

The system cannot handle open-ended or complex research queries such as:

- “What are the latest trends in this field?”
- “Which method performs better and why?”

It only processes the documents that are explicitly provided.

---

# Dataset-Related Limitations

The system uses the arXiv dataset from Kaggle, which provides metadata and abstracts instead of full research papers. This introduces additional constraints.

---

## 7. Abstract-Level Analysis Only

The analysis is performed using:

- title
- abstract

Full research paper content is not available, so the system cannot analyze:

- methodology
- experiments
- results
- detailed discussions

This limits the depth and accuracy of topic modeling.

---

## 8. Limited Effectiveness of Summarization

Abstracts are already short summaries of research papers.

Therefore:

- There is very little content available for further summarization
- The generated summaries may closely resemble the original abstract

---

## 9. No Citation or Research Relationship Analysis

The dataset does not include citation links between papers.

Because of this, the system cannot:

- identify influential papers
- track the evolution of a research field
- analyze connections between studies

---

## 10. Domain Imbalance

The arXiv dataset contains a higher number of papers in certain domains such as:

- Computer Science
- Physics

This may bias the topic modeling results toward these dominant fields.

---

## 11. Short and Noisy Text

Some abstracts are:

- very short
- highly technical
- filled with domain-specific symbols

This can reduce the effectiveness of:

- TF-IDF feature extraction
- clustering
- topic modeling

---

# Conclusion

These limitations highlight the gap between traditional NLP-based text analysis and intelligent research assistance.

They motivate the need for the Milestone-2 system, which will:

- retrieve real-time information
- understand context semantically
- perform multi-step reasoning
- generate structured research reports using agentic AI.