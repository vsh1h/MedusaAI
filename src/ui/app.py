import streamlit as st
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from PIL import Image

# Add project root to sys.path to resolve 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import existing NLP modules
from src.summarization.summarizer import split_sentences, generate_summary
from src.summarization.topic_inference import predict_topic, get_topic_distribution
from src.search.search_engine import search, summarize_topic

# Page Configuration
st.set_page_config(
    page_title="Research Topic Analysis System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Premium Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d0e12;
        color: #e0e0e0;
    }
    
    .main {
        background: radial-gradient(circle at top right, #1a1c24, #0d0e12);
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff9800, #f57c00);
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(245, 124, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 124, 0, 0.4);
    }
    
    .metric-card {
        background: rgba(38, 39, 48, 0.5);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: #ff9800;
    }
    
    .keyword-badge {
        display: inline-block;
        padding: 6px 16px;
        margin: 6px;
        background: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
        color: #ffb74d;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }

    .stTextInput>div>div>input {
        background-color: #1e1f26;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }

    .summary-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 16px;
        border-left: 4px solid #ff9800;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "results" not in st.session_state:
    st.session_state.results = {}

def extract_text(uploaded_file):
    """Extract text from uploaded PDF or TXT files."""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {e}")
    return None

def run_analysis(keywords_input, uploaded_files):
    all_text = ""
    doc_stats = []
    
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        if text:
            all_text += text + "\n"
            words = text.split()
            doc_stats.append({
                "name": uploaded_file.name,
                "word_count": len(words)
            })
    
    if not all_text:
        st.error("No text could be extracted from the uploaded files.")
        return

    # NLP Pipeline
    sentences = split_sentences(all_text)
    topic_id, keywords = predict_topic(all_text)
    topic_distribution = get_topic_distribution(all_text)
    
    # Get keyword scores from topics.json
    try:
        with open("data/processed/topics.json", "r") as f:
            topics_data = json.load(f)
            topic_key = f"topic_{topic_id}"
            keyword_scores = topics_data[topic_key].get("scores", [0.5] * len(keywords))
    except:
        keyword_scores = [0.5] * len(keywords)

    summary = generate_summary(
        sentences,
        keywords if keywords else [],
        topic_distribution,
        top_n=5
    )
    
    # Store results
    st.session_state.results = {
        "stats": {
            "num_docs": len(uploaded_files),
            "total_words": sum(s["word_count"] for s in doc_stats),
            "avg_words": sum(s["word_count"] for s in doc_stats) / len(doc_stats) if doc_stats else 0
        },
        "source_papers": [{"title": f.name, "score": 1.0} for f in uploaded_files],
        "keywords": keywords if keywords else (keywords_input.split() if keywords_input else []),
        "keyword_scores": keyword_scores,
        "topic_id": topic_id,
        "summary": summary,
        "topic_distribution": topic_distribution.tolist()
    }
    st.session_state.analyzed = True

def run_keyword_search(query):
    """Run analysis path based on knowledge base search."""
    results = search(query)
    
    if not results:
        st.error(f"No relevant papers found for '{query}' in the knowledge base.")
        return
        
    combined_text = " ".join([r["summary"] for r in results])
    prediction_text = f"{query} {combined_text}" # Include query to weight intent
    sentences = split_sentences(combined_text)
    topic_id, keywords = predict_topic(prediction_text)
    topic_distribution = get_topic_distribution(prediction_text)
    summary = summarize_topic(results)
    
    # Get keyword scores 
    try:
        with open("data/processed/topics.json", "r") as f:
            topics_data = json.load(f)
            topic_key = f"topic_{topic_id}"
            keyword_scores = topics_data[topic_key].get("scores", [0.5] * len(keywords))
    except:
        keyword_scores = [0.5] * len(keywords)

    st.session_state.results = {
        "stats": {
            "num_docs": len(results),
            "total_words": sum(len(r["summary"].split()) for r in results),
            "avg_words": sum(len(r["summary"].split()) for r in results) / len(results) if results else 0
        },
        "source_papers": [{"title": r["title"], "score": r["score"]} for r in results[:5]],
        "keywords": keywords,
        "keyword_scores": keyword_scores,
        "topic_id": topic_id,
        "summary": summary,
        "topic_distribution": topic_distribution.tolist()
    }
    st.session_state.analyzed = True

# Main Header Section
def show_input_page():
    # Use Container for padding and structure
    with st.container():
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("<h1 style='font-size: 3em; margin-bottom: 0;'>Input Section</h1>", unsafe_allow_html=True)
            st.write("Define your research scope and upload your documents for deep analysis.")
            st.write("")
            
            keywords = st.text_input("Research Topic Keywords", placeholder="e.g. quantum entanglement analysis")
            
            st.markdown("""
                <div style='display: flex; align-items: center; margin: 20px 0;'>
                    <div style='flex-grow: 1; height: 1px; background: rgba(255,255,255,0.1);'></div>
                    <span style='margin: 0 15px; color: rgba(255,255,255,0.4); font-size: 0.9em; font-weight: bold;'>OR</span>
                    <div style='flex-grow: 1; height: 1px; background: rgba(255,255,255,0.1);'></div>
                </div>
            """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Upload Research Papers", 
                type=["pdf", "txt"], 
                accept_multiple_files=True,
                help="Limit 200MB per file - PDF, TXT"
            )
            
            st.write("")
            if st.button("Run Analysis"):
                if keywords and not uploaded_files:
                    with st.spinner("Searching knowledge base..."):
                        run_keyword_search(keywords)
                        st.rerun()
                elif uploaded_files:
                    with st.spinner("Analyzing research papers..."):
                        run_analysis(keywords, uploaded_files)
                        st.rerun()
                else:
                    st.warning("Please provide keywords or upload at least one paper.")

        with col2:
            # Load generated header image
            header_img_path = "/Users/kirtig/.gemini/antigravity/brain/a61ef82e-815f-4978-82a0-5d2805dfbafa/research_analysis_header_1771855161620.png"
            if os.path.exists(header_img_path):
                st.image(header_img_path, width='stretch')
            else:
                st.info("Header image not found.")

def show_results_page():
    res = st.session_state.results
    
    # Load semantic labels from topics.json
    try:
        with open("data/processed/topics.json", "r") as f:
            topics_data = json.load(f)
            semantic_labels = topics_data.get("semantic_labels", {})
            topic_words = {k: v["words"] for k, v in topics_data.items() if k.startswith("topic_")}
    except:
        semantic_labels = {}
        topic_words = {}

    st.title("Analysis Results")
    if st.button("Back to Input"):
        st.session_state.analyzed = False
        st.rerun()
        
    st.write("### Document Statistics")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card">Number of Documents<br><span style="font-size: 2em; color: #ff9800;">{res["stats"]["num_docs"]}</span></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card">Total Words<br><span style="font-size: 2em; color: #ff9800;">{res["stats"]["total_words"]:,}</span></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card">Average Words/Doc<br><span style="font-size: 2em; color: #ff9800;">{int(res["stats"]["avg_words"]):,}</span></div>', unsafe_allow_html=True)
        
    st.write("")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("### Extracted Keywords")
        st.markdown("<div style='margin-bottom: 20px;'>" + " ".join([f'<span class="keyword-badge">{k}</span>' for k in res["keywords"]]) + "</div>", unsafe_allow_html=True)
        
    with c2:
        st.write("### Identified Topics")
        topic_key = f"topic_{res['topic_id']}"
        topic_name = semantic_labels.get(topic_key, f"Topic {res['topic_id']}")
        keywords_str = ", ".join(topic_words.get(topic_key, [])[:5])
        
        st.markdown(f'''
            <div class="metric-card" style="border-left: 5px solid #4CAF50;">
                Predominant Topic: <span style="color: #4CAF50; font-size: 1.5em; font-weight: bold;">{topic_name}</span><br>
                <span style="font-size: 0.9em; opacity: 0.8;">Key terms: {keywords_str}</span>
                <div style="font-size: 0.7em; opacity: 0.5; margin-top: 10px;">Debug ID: {res['topic_id']}</div>
            </div>
        ''', unsafe_allow_html=True)

    st.write("")
    st.write("### Analysis Summary")
    st.markdown(f'<div class="summary-box">{res["summary"]}</div>', unsafe_allow_html=True)
    
    if "source_papers" in res and res["source_papers"]:
        with st.expander("Analyzed Source Materials"):
            for paper in res["source_papers"]:
                st.write(f"- **{paper['title']}** (Relevance: {paper['score']})")
    
    st.write("")
    st.write("### Visualizations")
    
    v1, v2 = st.columns([1, 1], gap="medium")
    with v1:
        st.write("#### Keyword Importance")
        importance_df = pd.DataFrame({
            "Keyword": res["keywords"][:10],
            "Importance": res["keyword_scores"][:10]
        })
        st.bar_chart(importance_df, x="Keyword", y="Importance", color="#ff9800")
        
    with v2:
        st.write("#### Topic Probability Distribution")
        
        labels = [semantic_labels.get(f"topic_{i}", f"Topic {i}") for i in range(len(res["topic_distribution"]))]
        
        dist_df = pd.DataFrame({
            "Topic": labels,
            "Probability": res["topic_distribution"]
        })
        st.bar_chart(dist_df, x="Topic", y="Probability", color="#2196F3")


# Router
if st.session_state.analyzed:
    show_results_page()
else:
    show_input_page()
