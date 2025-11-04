# chain.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


def make_rag_chain(retriever):
    """
    RAG Chain for YouTube Chatbot (Gemini + FAISS)
    Compatible with:
      - google-generativeai >= 0.7.2
      - langchain-google-genai >= 1.0.3
      - langchain-core >= 0.2.x
    """

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # use exactly the name from the list you get
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


    # Prompt Template
    prompt_template = """
You are a helpful AI assistant that summarizes and answers questions 
based ONLY on the YouTube transcript context provided below.

Transcript Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )

    def chain(query):
        """Retrieve relevant chunks and generate a Gemini response."""
        # Fetch relevant transcript pieces
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Ask Gemini
        result = llm.invoke(prompt.format(context=context, question=query))
        return result.content

    return chain
