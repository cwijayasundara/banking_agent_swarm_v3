from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pdf_parser import parse_pdf_file
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ = load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

file_paths = ["docs/interest-rates_1.pdf", 
              "docs/interest-rates_2.pdf", 
              "docs/interest-rates_3.pdf"]

VECTOR_STORE_PATH= "./vector_store"
VECTOR_STORE_NAME = "bank_accounts"

vector_store = Chroma(embedding_function=embeddings, 
                      persist_directory=VECTOR_STORE_PATH,
                      collection_name=VECTOR_STORE_NAME)

def persist_documents_to_vector_store(file_paths):
    """Function to persist documents to vector store"""

    # Parse the PDF files
    documents = []
    for file_path in file_paths:
        documents.extend(parse_pdf_file(file_path))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    # Index chunks
    _ = vector_store.add_documents(documents=splits)

#Persist the documents to the vector store
persist_documents_to_vector_store(file_paths)