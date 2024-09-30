from langchain_chroma import Chroma

import ollama
from langchain_ollama import OllamaEmbeddings

# open Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="/opt/support-issues/chromadb")

# Call Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    print("formatted prompt:")
    print(formatted_prompt)
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# RAG Setup
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    print("number of retrieved documents: %i", len(retrieved_docs))
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Use the RAG App
Question = "Summarize issue 2925 for Vodacom"
result = rag_chain(f"{Question}")
print(f"Question : {Question}")
print(f"Response : {result}")
