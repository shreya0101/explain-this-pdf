import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re

# Load models
import os
from sentence_transformers import SentenceTransformer

# Ensure model is loaded from cache
os.environ["TRANSFORMERS_CACHE"] = "./cache"
embedder = SentenceTransformer('all-MiniLM-L6-v2')

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

st.set_page_config(page_title="Explain This PDF", layout="wide")
st.title("🧠 Explain This PDF")

uploaded_files = st.file_uploader(
    "📄 Upload one or more PDF documents (any topic)",
    type="pdf",
    accept_multiple_files=True
)

all_paragraphs = []
texts, embeddings = [], None
chat_history = []

# --------- Function: Extract text by paragraphs ---------
def extract_text_with_pages(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    paragraphs = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text()
        paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
        for p in paras:
            paragraphs.append({
                'text': p,
                'page': page_num,
                'filename': file.name
            })
    return paragraphs

# --------- Function: Highlight answer in context ---------
def highlight_answer_in_context(answer, context):
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    return pattern.sub(f"**{answer}**", context)

# --------- Function: Ask a question and return best match ---------
def ask_question(query):
    if not texts or embeddings is None:
        return "Please upload and index PDFs first.", "", {}

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_idx = scores.argmax().item()
    chunk = all_paragraphs[best_idx]

    try:
        result = qa_pipeline(question=query, context=chunk["text"])
        answer = result["answer"]
        highlighted_context = highlight_answer_in_context(answer, chunk["text"])
    except Exception:
        answer = "Sorry, I couldn't find an answer."
        highlighted_context = chunk["text"]

    chat_history.append((query, answer))
    return answer, highlighted_context, chunk

# --------- Process uploaded PDFs ---------
if uploaded_files:
    all_paragraphs = []
    for file in uploaded_files:
        all_paragraphs += extract_text_with_pages(file)
    texts = [p['text'] for p in all_paragraphs]
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    chat_history.clear()
    st.success(f"✅ Successfully indexed {len(texts)} chunks from {len(uploaded_files)} PDF(s).")

# --------- Question input + answer display ---------
query = st.text_input("🔍 Ask a question about your uploaded PDFs")

if query:
    answer, highlighted_context, meta = ask_question(query)

    st.markdown(f"### 🧠 Answer:\n**{answer}**")

    with st.expander("📌 Full Context with Highlight"):
        st.markdown(highlighted_context)
        if meta:
            st.caption(f"📄 From: `{meta['filename']}`, Page {meta['page']}")

