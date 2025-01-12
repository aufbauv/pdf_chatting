import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Storing api key as secret during deployment
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():
    prompt_template = """
    You are an assistant that provides detailed and accurate answers based on the given context. If the answer is not available in the context, say: "Answer is not available in the context."

    Previous conversation history:
    {context}

    Current question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # Load the vector store for document retrieval
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Build the context from the chat history
    if "chat_history" in st.session_state:
        context = "\n".join([f"{msg['role']}: {msg['text']}" for msg in st.session_state.chat_history])
    else:
        context = ""

    # Add the current user question to the context
    context += f"\nUser: {user_question}"

    chain = get_conversational_chain()

    try:
        # Generate the response using the current context and question
        response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
        answer = response["output_text"]

        # Append the current user question and bot answer to the chat history
        st.session_state.chat_history.append({"role": "User", "text": user_question})
        st.session_state.chat_history.append({"role": "Bot", "text": answer})

        # Display the entire conversation history
        for message in st.session_state.chat_history:
            if message["role"] == "User":
                st.markdown(f'<p style="color: red; font-weight: bold;">You:</p> {message["text"]}', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color: orange; font-weight: bold;">File Crawler:</p> {message["text"]}', unsafe_allow_html=True)

    except Exception as e:
        # Simple error handling for 429 ResourceExhausted
        if "ResourceExhausted" in str(e) or "429" in str(e):
            st.error("API quota exhausted. Please try again later.")
        else:
            # Handle any other errors
            st.error(f"An error occurred: {str(e)}")



def main():
    st.set_page_config("PDF Chatting")
    st.header("File Crawler 🤖")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask Questions from the uploaded PDF Files")

    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("📄 Uploader:")
        pdf_docs = st.file_uploader("PDFs Only", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Scanning..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        st.markdown('<h2 style="color: orange;">About:</h2>', unsafe_allow_html=True)
        st.write("File Crawler allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.")


if __name__ == "__main__":
    main()