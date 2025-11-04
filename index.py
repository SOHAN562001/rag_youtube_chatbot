# index.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def build_faiss_index(transcript_text: str):
    """
    Builds a FAISS retriever index from YouTube transcript text.
    Returns a retriever object for RAG search.
    """
    # ---- Split transcript into smaller chunks ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(transcript_text)
    print(f"✅ Split transcript into {len(chunks)} chunks")

    # ---- Create sentence embeddings ----
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ---- Build FAISS vector index ----
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("✅ FAISS vector index built successfully!")
    return retriever
