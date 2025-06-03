import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredXMLLoader, UnstructuredPowerPointLoader
)
from dotenv import load_dotenv
import tempfile
import time
from fpdf import FPDF
import speech_recognition as sr
import openai

# Load environment
load_dotenv()

# Set up API keys
os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Knowva", page_icon="🤖", layout="centered")

# Inject custom CSS with enhanced design and robot toy element
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css-"], .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #87CEEB 0%, #FFB6C1 50%, #FFD700 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientShift 15s ease infinite !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 100vh !important;
        font-family: 'Poppins', sans-serif !important;
        color: #ffffff !important;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: transparent !important;
    }
    .block-container {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px;
        padding: 2rem !important;
        max-width: 900px;
        margin: 2rem auto !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #333333 !important;
    }
    .sticky-header {
        position: sticky;
        top: 0;
        background: linear-gradient(90deg, #87CEEB, #FFD700) !important;
        padding: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        z-index: 999 !important;
        text-align: center !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: transparent !important;
        background-clip: text !important;
        -webkit-background-clip: text !important;
        border-radius: 0 0 15px 15px !important;
    }
    .footer {
        text-align: center !important;
        padding: 1rem !important;
        margin-top: 3rem !important;
        font-size: 0.8rem !important;
        color: #e0e0e0 !important;
        background: linear-gradient(90deg, rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0.3)) !important;
        border-radius: 10px !important;
    }
    .chat-bubble {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        padding: 1rem !important;
        border-radius: 15px !important;
        margin: 1rem 0 !important;
        animation: fadeIn 0.6s ease-in !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        color: #333333 !important;
        transition: transform 0.3s ease !important;
    }
    .chat-bubble:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15) !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    @keyframes float {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    .stButton>button {
        background: linear-gradient(90deg, #ffd500, #ffab00) !important;
        color: #1e3c72 !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2), 0 0 10px #ffd500 !important;
        background: linear-gradient(90deg, #ffab00, #ffd500) !important;
    }
    .hero-box {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        padding: 3rem 2rem !important;
        text-align: center !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    .hero-box h1 {
        font-size: 3.5rem !important;
        background: linear-gradient(90deg, #1e3c72, #2a5298) !important;
        background-clip: text !important;
        -webkit-background-clip: text !important;
        color: transparent !important;
        animation: bounce 2s infinite !important;
    }
    .hero-box h3 {
        background: linear-gradient(90deg, #2a5298, #1e3c72) !important;
        background-clip: text !important;
        -webkit-background-clip: text !important;
        color: transparent !important;
        font-weight: 600 !important;
    }
    .hero-box p {
        color: #333333 !important;
        font-size: 1.1rem !important;
    }
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #87CEEB, #FFD700) !important;
        border-radius: 10px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(90deg, #FFD700, #87CEEB) !important;
    }
    /* Style for text input and selectbox */
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        color: #333333 !important;
    }
    div[data-baseweb="select"] > div:hover {
        background: rgba(255, 255, 255, 0.3) !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        color: #333333 !important;
        padding: 0.5rem !important;
    }
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 0 8px #ffd500 !important;
    }
    /* Robot toy element */
    .robot-toy {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        font-size: 3rem !important;
        animation: float 3s ease-in-out infinite !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.8), 0 0 20px rgba(255, 215, 0, 0.5) !important;
        z-index: 1000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Session setup
if "started" not in st.session_state:
    st.session_state.started = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Dark mode toggle function
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply dark mode styles
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        html, body, [class*="css-"], .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 50%, #e74c3c 100%) !important;
            background-size: 200% 200% !important;
            animation: gradientShift 15s ease infinite !important;
        }
        .block-container, .hero-box {
            background: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(10px) !important;
            color: #ffffff !important;
        }
        .chat-bubble {
            background: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(8px) !important;
            color: #ffffff !important;
        }
        .chat-bubble:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #e67e22, #d35400) !important;
            color: #ffffff !important;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #d35400, #e67e22) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2), 0 0 10px #e67e22 !important;
        }
        .hero-box h1 { 
            background: linear-gradient(90deg, #ecf0f1, #bdc3c7) !important;
            background-clip: text !important;
            -webkit-background-clip: text !important;
            color: transparent !important;
        }
        .hero-box h3 { 
            background: linear-gradient(90deg, #bdc3c7, #ecf0f1) !important;
            background-clip: text !important;
            -webkit-background-clip: text !important;
            color: transparent !important;
        }
        .hero-box p { color: #ecf0f1 !important; }
        .sticky-header { 
            background: linear-gradient(90deg, #2c3e50, #e74c3c) !important;
            color: transparent !important;
            background-clip: text !important;
            -webkit-background-clip: text !important;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(90deg, #3498db, #e74c3c) !important;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(90deg, #e74c3c, #3498db) !important;
        }
        div[data-baseweb="select"] > div {
            background: rgba(0, 0, 0, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: #ffffff !important;
        }
        div[data-baseweb="select"] > div:hover {
            background: rgba(0, 0, 0, 0.7) !important;
        }
        .stTextInput > div > div > input {
            background: rgba(0, 0, 0, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: #ffffff !important;
        }
        .stTextInput > div > div > input:focus {
            background: rgba(0, 0, 0, 0.7) !important;
            box-shadow: 0 0 8px #e67e22 !important;
        }
        .robot-toy {
            text-shadow: 0 0 10px rgba(231, 76, 60, 0.8), 0 0 20px rgba(231, 76, 60, 0.5) !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Add the robot toy element
st.markdown("<div class='robot-toy'>🤖</div>", unsafe_allow_html=True)

# Hero screen
if not st.session_state.started:
    st.markdown("""
        <div class='hero-box'>
            <h1>Meet Knowva 🤖</h1>
            <h3>Your AI Knowledge Companion</h3>
            <p>Get instant insights from your documents, simplify medical reports, and ask anything — all in one smart, secure AI experience.</p>
            <form action="">
                <button style='background: linear-gradient(90deg, #ffd500, #ffab00);border:none;padding:1rem 3rem;border-radius:50px;font-size:1.2rem;font-weight:600;cursor:pointer;color:#1e3c72;'>Get Started</button>
            </form>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Continue to App", key="start"):
        st.session_state.started = True
    st.stop()

# Sticky header
st.markdown("<div class='sticky-header'>🤖 Knowva Assistant</div>", unsafe_allow_html=True)

# Dark mode toggle button
st.button("Toggle Dark Mode", on_click=toggle_dark_mode, key="dark_mode_toggle")

# Interface
mode = st.selectbox("Document Type", ["General", "Medical"], key="mode_select")
model_choice = st.selectbox("Model", ["NVIDIA LLaMA3", "OpenAI GPT-4"], key="model_select")
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "docx", "xml", "ppt", "pptx"], accept_multiple_files=True, key="file_uploader")

if uploaded_files:
    st.markdown("#### 💡 Sample Questions")
    if mode == "Medical":
        st.markdown("- What does this report indicate?\n- Explain these lab results in simple terms")
    else:
        st.markdown("- What are the main topics covered?\n- Summarize this document for me")

# Prompts
general_prompt = ChatPromptTemplate.from_template("""
Use the provided context to answer the question. If the context is insufficient,
include your best possible answer using general knowledge.
<context>
{context}
</context>
Question: {input}
""")

medical_prompt = ChatPromptTemplate.from_template("""
You are a medical assistant AI. Your job is to explain medical reports in clear,
simple language for a non-medical person. Use the provided report context and give
accurate, patient-friendly explanations.
<context>
{context}
</context>
Question: {input}
""")

# Load and process documents with error handling
def load_file(uploaded_file):
    try:
        suffix = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        loaders = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "docx": UnstructuredWordDocumentLoader,
            "xml": UnstructuredXMLLoader,
            "ppt": UnstructuredPowerPointLoader,
            "pptx": UnstructuredPowerPointLoader,
        }
        loader = loaders.get(suffix, lambda x: [])
        return loader(tmp_path).load()
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        return []

def process_files(files):
    try:
        all_docs = []
        progress_bar = st.progress(0)
        total_files = len(files) if files else 1  # Avoid division by zero
        for i, file in enumerate(files):
            docs = load_file(file)
            if docs:
                all_docs.extend(docs)
            progress_bar.progress((i + 1) / total_files)
        if not all_docs:
            st.error("No valid documents loaded. Please check the file formats.")
            return
        chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50).split_documents(all_docs)
        embeddings = NVIDIAEmbeddings()
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
        st.session_state.ready = True
        st.success("✅ Documents embedded successfully!")
        progress_bar.empty()
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        progress_bar.empty()

if uploaded_files and st.button("📚 Scan Documents", key="scan_button"):
    process_files(uploaded_files)

# Voice or text input with error handling
use_voice = st.checkbox("🎙️ Use voice input", key="voice_check")
query = ""
if use_voice:
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Speak now...")
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
    except sr.UnknownValueError:
        st.error("Could not understand audio. Please try again.")
    except sr.RequestError as e:
        st.error(f"Voice recognition service error: {str(e)}")
    except Exception as e:
        st.error(f"Voice recognition error: {str(e)}")
else:
    with st.expander("💡 Sample Questions", expanded=False):
        if mode == "Medical":
            st.markdown("- What does this report indicate?\n- Explain these lab results in simple terms")
        else:
            st.markdown("- What are the main topics covered?\n- Summarize this document for me")
    query = st.text_input("💬 Ask a question:", key="query_input")

# Run query with error handling
if query and "vectors" in st.session_state:
    try:
        retriever = st.session_state.vectors.as_retriever()
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct") if model_choice == "NVIDIA LLaMA3" else ChatOpenAI(model_name="gpt-4")
        prompt = medical_prompt if mode == "Medical" else general_prompt
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

        with st.spinner("Processing your request... 🤖"):
            start = time.process_time()
            response = chain.invoke({"input": query})
            st.session_state.chat_history.append((query, response['answer']))
            st.success("🧠 Answered!")
            st.caption(f"⏱️ {round(time.process_time() - start, 2)} seconds")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

# Display chat history
with st.expander("View Chat History", expanded=True):
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"<div class='chat-bubble'><b>❓ Q{i}:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble'><b>🧠 A{i}:</b> {a}</div>", unsafe_allow_html=True)

# PDF export with error handling
if st.session_state.chat_history:
    def export_chat_to_pdf(chat_history):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Chat Summary", ln=True, align='C')
            pdf.ln(10)
            for i, (q, a) in enumerate(chat_history, 1):
                pdf.multi_cell(0, 10, f"Q{i}: {q}", align='L')
                pdf.multi_cell(0, 10, f"A{i}: {a}", align='L')
                pdf.ln(5)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf.output(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            return None

    if st.button("📥 Download Your Chat as PDF", key="download_button"):
        pdf_path = export_chat_to_pdf(st.session_state.chat_history)
        if pdf_path:
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="Click here to download your chat summary",
                    data=file,
                    file_name="chat_summary.pdf",
                    mime="application/pdf"
                )

# Footer
st.markdown("<div class='footer'>© 2025 Knowva | Built with LangChain + NVIDIA AI</div>", unsafe_allow_html=True)