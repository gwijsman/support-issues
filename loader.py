from langchain_community.document_loaders import TextLoader
#import bs4

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

#import ollama
from langchain_ollama import OllamaEmbeddings

textfile = '/opt/support-issues/output/2925.txt'

# Load the data
loader = TextLoader(
    file_path=(f"{textfile}")
#    bs_kwargs=dict(
#        parse_only=bs4.SoupStrainer(
#            class_=("td-post-content tagdiv-type", "td-post-header", "td-post-title")
#        )
#    ),
)
docs = loader.load()
print("loaded document...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
print("defined text splitter...")
splits = text_splitter.split_documents(docs)
print("splitted the documents...")
print(" number: %i", len(splits))

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")
print("created the embeddings...")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="/opt/support-issues/chromadb")

