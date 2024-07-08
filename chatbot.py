import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")


#upload pdf
st.header("My First Chatbot")

with st.sidebar:
    st.title("Load Documents")
    file = st.file_uploader("Upload a Pdf and start asking questions", type="pdf")

#Extract text
if file:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    #Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    #generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector stores - FAISS
    vector = FAISS.from_texts(chunks, embeddings)

    #get user question
    question = st.text_input("Type your question here: ")

    #do similarity search
    if question:
        matches = vector.similarity_search(question)
        # st.write(matches)

        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens= 1000,
            model_name="gpt-3.5-turbo"
        )

        #output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matches,question=question)
        st.write(response)