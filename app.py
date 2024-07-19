from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models.openai import ChatOpenAI
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import os

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

pinecone_api_key = os.getenv('pinecone_api_key')
sec_key=os.getenv('openai_api_key')


pc = Pinecone(api_key=pinecone_api_key)


index_name = "langchain-index"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

index = pc.Index(index_name)

def retrieve_results():

    if 'vectors' not in st.session_state:
        st.session_state.loader = PyPDFLoader('business/Business Management And Organization Booklet.pdf')
        st.session_state.docs = st.session_state.loader.load() 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200 )
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.embeddings = OpenAIEmbeddings(api_key=sec_key)
        st.session_state.vectors = PineconeVectorStore.from_documents(st.session_state.documents,st.session_state.embeddings,index_name=index_name)
        
st.title('RAG Business Chatbot application')

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", api_key=sec_key)

prompts = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}""")

prompt = st.text_input("Enter your question from documents")

if st.button("Retrieve result"):
    retrieve_results()
    st.write("Vector Store DB Is Read")

if prompt:
    document_chain = create_stuff_documents_chain(llm,prompts)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    response = retriever_chain.invoke({"input":prompt})
    st.write(response['answer'])
