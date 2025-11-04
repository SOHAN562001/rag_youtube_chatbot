import os
import streamlit as st
from dotenv import load_dotenv

from loader import extract_video_id, fetch_transcript, transcript_to_text
from index import build_faiss_index
from chain import make_rag_chain

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="RAG-Based YouTube Chatbot",
    page_icon="🎬",
    layout="centered",
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🎥 RAG-Based YouTube Chatbot")
st.markdown(
    """
    <div style='text-align:center;'>
    <b>Built using LangChain · FAISS · Google Gemini</b><br>
    Developed by <b>Sohan Ghosh</b> | MSc Data Science & AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------------------------------
url = st.text_input("🎦 Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=abc123xyz")

if not url:
    st.info("👆 Paste a YouTube video link above to begin.")
    st.stop()

# -----------------------------------------------------------------------------
# BUILD KNOWLEDGE BASE
# -----------------------------------------------------------------------------
if st.button("🧠 Build Knowledge Base"):
    with st.spinner("⏳ Fetching transcript & building FAISS index..."):
        try:
            video_id = extract_video_id(url)
            transcript_data = fetch_transcript(video_id)
            transcript_text = transcript_to_text(transcript_data)

            st.success(f"✅ Transcript extracted successfully ({len(transcript_text.split())} words)")

            retriever = build_faiss_index(transcript_text)
            st.session_state.retriever = retriever

            st.success("✅ Knowledge Base built successfully!")
            st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", width=480)

        except Exception as e:
            st.error(f"❌ Error while building knowledge base: {e}")

# -----------------------------------------------------------------------------
# Q&A / SUMMARIZATION SECTION
# -----------------------------------------------------------------------------
if "retriever" in st.session_state:
    st.markdown("### 💬 Ask about the video")
    query = st.text_input("Your question:", placeholder="e.g., What are the main points discussed?")

    col1, col2 = st.columns(2)
    ask_btn = col1.button("🔍 Ask Question")
    summarize_btn = col2.button("🧾 Summarize Video")

    if ask_btn or summarize_btn:
        with st.spinner("💡 Gemini is thinking..."):
            try:
                retriever = st.session_state.retriever
                rag_chain = make_rag_chain(retriever)

                if summarize_btn:
                    query = "Summarize this YouTube video clearly in 4-5 bullet points."

                # ✅ Direct function call (no .run)
                answer = rag_chain(query)

                st.markdown("### 🧠 Gemini’s Response")
                st.write(answer)

            except Exception as e:
                st.error(f"⚠️ Gemini error: {e}")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px;'>
        <b>RAG-Based YouTube Chatbot</b> · Powered by <b>LangChain</b>, <b>FAISS</b> & <b>Google Gemini</b><br>
        © Developed by <b>Sohan Ghosh</b> | MSc Data Science & AI
    </div>
    """,
    unsafe_allow_html=True
)
