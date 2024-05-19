# openai api key

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create a PyPDFDirectoryLoader
pdf_folder = '.\\samplepdf\\'
loader = PyPDFDirectoryLoader(pdf_folder)

# Create a custom text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
)

# Load and split the documents
documents = loader.load_and_split(text_splitter)

# embedding 
persist_directory = 'db'
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=persist_directory)

vectorstore.persist() #save the vectorstore to chromadb
