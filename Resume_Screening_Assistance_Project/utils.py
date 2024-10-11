from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub

import time


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#Function to push data to Vector Store - Pinecone here
import pinecone
from langchain.vectorstores import Pinecone
from pinecone import ServerlessSpec
import os

# Function to push data to Vector Store - Pinecone here
# Updated imports based on reference
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone  # Updated import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import time
import joblib

# Function to push data to Vector Store - Pinecone
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    # Initialize Pinecone client
    print(pinecone_apikey)
    pc = PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)

    # Check if the index already exists, create if it doesn't
    if pinecone_index_name not in pc.list_indexes().names():
        # Create a new index if it doesn't exist
        pc.create_index(
            name=pinecone_index_name,
            dimension=embeddings.embed_query_dimension(),  # Ensure this matches the dimension of the embedding model
            metric='cosine'  # Similarity metric
        )

    # Get the index
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)

    # Now push the documents to Pinecone index using Langchain Pinecone wrapper
    index.add_documents(docs)


# Function to pull information from Vector Store - Pinecone
def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    # Optional delay to handle free-tier limits
    print("20secs delay...")
    time.sleep(20)

    # Initialize Pinecone client
    pc = PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)

    # Retrieve the existing index
    if pinecone_index_name in pc.list_indexes().names():
        index = Pinecone.from_existing_index(pinecone_index_name, embeddings)  # Get the index
        return index
    else:
        raise ValueError(f"Index {pinecone_index_name} not found.")


# Function to get relevant documents from the vector store based on user input
def similar_docs(query, k, pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, unique_id):
    # Initialize Pinecone client
    pc = PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)

    # Pull index from Pinecone
    index = pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings)

    # Perform similarity search
    similar_docs = index.similarity_search(
        query=query,
        k=int(k),
        filter={"unique_id": unique_id}
    )
    return similar_docs


# Function to create embeddings instance
def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Function to get a summary or an answer from documents
def get_summary_or_answer(docs, user_input):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response


# Example prediction function (if using a machine learning model)
def predict(query_result):
    model = joblib.load('modelsvm.pk1')  # Assuming a pre-trained model exists
    result = model.predict([query_result])
    return result[0]


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary




    