RAG-Based YouTube Chatbot (Local FAISS Version)

Developed by: Sohan Ghosh
Program: MSc Data Science & AI
Built using: LangChain · FAISS · Google Gemini

1. Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can intelligently answer questions about any YouTube video by leveraging its transcript.

The application uses LangChain for orchestration, FAISS for vector similarity search, HuggingFace for embeddings, and Google Gemini for response generation — all combined in a user-friendly Streamlit interface.

This implementation runs fully on local setup — ideal for academic or research use without requiring cloud deployment.

2. Key Features

Automatic extraction of YouTube video transcript

Text chunking using RecursiveCharacterTextSplitter

Embeddings generation using HuggingFace MiniLM-L6-v2

FAISS-based vector indexing and retrieval

Context-enriched responses generated via Gemini API

Clean and minimal Streamlit interface

3. Technology Stack
Component	Library
LLM	Google Gemini (langchain-google-genai)
Embeddings	HuggingFace Sentence Transformers
Vector Store	FAISS
Framework	LangChain
Frontend	Streamlit
Data Source	YouTube Transcript API
4. Project Structure
rag_youtube_chatbot/
│
├── app.py              # Streamlit interface
├── chain.py            # RAG Chain logic (Retriever + Gemini)
├── index.py            # FAISS index creation & management
├── loader.py           # YouTube transcript fetching utilities
├── requirements.txt    # Python dependencies
├── .env                # Gemini API Key (not uploaded)
└── assets/             # Screenshots for documentation

5. How to Run Locally
Step 1. Clone Repository
git clone https://github.com/SOHAN562001/rag_youtube_chatbot.git
cd rag_youtube_chatbot

Step 2. (Optional) Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

Step 3. Install Dependencies
pip install -r requirements.txt

Step 4. Set API Key

Create a .env file in the project root with:

GOOGLE_API_KEY=your_gemini_api_key

Step 5. Run the Application
streamlit run app.py


Access it locally at:
http://localhost:8501

6. Demonstration Screenshots
(a) Transcript Extraction and Knowledge Base Building

(b) Asking Contextual Questions from Video

(c) Gemini-Generated Contextual Summaries

7. Core Workflow

Extracts transcript text from YouTube video

Splits text into manageable chunks

Generates vector embeddings using MiniLM

Builds FAISS index for similarity search

On query, retrieves top-relevant chunks

Sends context and query to Gemini for final response

8. Example Code Snippet
query = "Explain Retrieval Augmented Generation in simple terms"
response = chain.invoke({"question": query})
print(response)


Output Example:

RAG (Retrieval Augmented Generation) combines retrieval and generation.
It fetches the most relevant transcript segments from FAISS and provides an accurate, context-driven response using Gemini.

9. Dependencies
streamlit
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-google-genai
google-generativeai
faiss-cpu
python-dotenv
yt-dlp
youtube-transcript-api
tqdm

10. Notes

.env file excluded via .gitignore

No ChromaDB used; fully FAISS-based

Works only for videos with transcripts enabled

Runs 100% locally — no cloud services required

11. Credits

LangChain

Google Gemini

HuggingFace Transformers

FAISS (Facebook AI Similarity Search)

Streamlit

YouTube Transcript API
