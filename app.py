import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Streamlit page settings
st.set_page_config(page_title="ChatBot", page_icon="🤖")

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Semantic Chunking
def semantic_chunk_text(text, max_chunk_size=1000):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Create and Save Vector Store
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational Chain
def get_conversational_chain():
    prompt_template = """
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    conversational_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    return load_qa_chain(conversational_model, chain_type="stuff", prompt=prompt)

# Main Streamlit Application
def main():
    st.title("🤖 ChatBot")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Welcome message
    if "welcome_message" not in st.session_state:
        st.session_state.welcome_message = False
        st.session_state.chat_history.append({"role": "assistant", "content": "Welcome! Upload a PDF and ask questions about its content and say Thank you to end the chat."})

    # Display existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # PDF Upload Section
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)
    if st.sidebar.button("Process PDFs"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(uploaded_files)
            if not raw_text:
                st.error("No text extracted. Please upload a valid PDF.")
                return
            text_chunks = semantic_chunk_text(raw_text)
            get_vector_store(text_chunks)
            st.session_state.pdf_processed = True
            st.success("PDF processed successfully!")
            st.session_state.chat_history.append({"role": "assistant", "content": "PDF processed successfully! You can now ask questions."})

    # Handle user input
    user_query = st.chat_input("Ask me anything...")
    if user_query:
        # Append user's message to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Check if PDF is processed
        if "pdf_processed" in st.session_state:
            # Load FAISS and get matching context
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_query)

            # Get response from the conversational chain
            chain = get_conversational_chain()
            response = chain.run(input_documents=docs, question=user_query)

            # Display response
            assistant_response = response.strip() if response else "I'm sorry, I couldn't find relevant information."
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "Please upload and process a PDF before asking questions."})
            st.chat_message("assistant").markdown("Please upload and process a PDF before asking questions.")

    # Refresh button to reset the chat
    if st.button("Refresh Chat"):
        st.session_state.chat_history = []
        st.session_state.welcome_message = False
        st.rerun()

if __name__ == "__main__":
    main()
