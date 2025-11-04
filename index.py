# index.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_faiss_index(transcript_text):
    """
    Build a vector index for the given transcript text.
    Uses Chroma (pure Python, Streamlit Cloud–compatible) instead of FAISS.
    """
    # Step 1: Split the transcript into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    docs = text_splitter.create_documents([transcript_text])

    # Step 2: Convert text chunks into embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Create a Chroma-based vectorstore
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Step 4: Return retriever object for similarity search
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever
