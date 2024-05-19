# openai api key
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
#import openai

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# load vectorstore from chromadb
vectorstore = Chroma(persist_directory='db', embedding_function=OpenAIEmbeddings())


retriever = vectorstore.as_retriever()

''' 
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 


Question: {question} 
Context: {context} 
Answer:
"""
'''

template = """You are an expert researcher and writer, tasked with answering any question.
You must only use information from the retrieved context to answer the question. 
Don't try to make up an answer. If you don't know the answer, just say that you don't know. 
Use an unbiased and journalistic tone. 
Give citation for the answer using [].
Put citations where they apply rather than putting them all at the end.
At the end of your answer, create a Reference section to list the citation source using retrieve document metadata. 


Question: {question} 
Context: {context} 
Answer:
"""




prompt = ChatPromptTemplate.from_template(template)

Chat_model = ChatOpenAI(openai_api_key = api_key)
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | Chat_model
    | StrOutputParser() 
)

