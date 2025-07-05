import streamlit as st
from pdf_reader import extract_text_from_pdf
from qa_engine import split_text, create_vector_store, get_answer
import tempfile
from PIL import Image
import base64

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stFileUploader>div>div>button {
        border-radius: 10px;
    }
    .stSlider>div>div>div>div {
        color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .css-1d391kg {
        padding-top: 1.5rem;
    }
    .answer-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    .preview-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📚 PDF Q&A Assistant")
    st.markdown("""
    ### About
    Upload a PDF document and ask questions about its content.
    The AI will analyze the text and provide answers.
    """)
    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Upload a PDF file
    2. Wait for text extraction
    3. Ask your question
    4. Get instant answers!
    """)
    st.markdown("---")
    st.markdown("""
    Made with ❤️ using [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), and [HuggingFace](https://huggingface.co)
    """)

# Main content
st.title("📚 PDF Q&A Assistant")
st.markdown("Extract insights from your PDF documents with AI-powered Q&A")

# File uploader with better styling
uploaded_file = st.file_uploader(
    "**Upload your PDF file here**",
    type="pdf",
    help="Supported formats: PDF",
    key="file_uploader"
)

if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Extract text
        text = extract_text_from_pdf(tmp_path)
        
        # Preview section
        with st.expander("📄 Extracted Text Preview", expanded=False):
            st.markdown(f'<div class="preview-box">{text[:500]}...</div>', unsafe_allow_html=True)
        
        # Split into chunks
        chunks = split_text(text)
        st.success(f"✅ Successfully extracted {len(chunks)} text chunks from the document.")

        # User sets chunk limit
        st.markdown("### 🔧 Performance Settings")
        max_chunks = st.slider(
            "Adjust the number of text chunks to use (more chunks = more accurate but slower):",
            100, 2000, 500, 100,
            help="Limiting chunks can improve response speed"
        )
        
        if len(chunks) > max_chunks:
            st.warning(f"⚠️ Using only first {max_chunks} chunks for faster processing.")
            chunks = chunks[:max_chunks]

        if len(chunks) == 0:
            st.error("No valid text found in this PDF. Please try another document.")
        else:
            # Build vector store
            with st.spinner("🔍 Creating search index..."):
                store = create_vector_store(chunks)
                st.success("✅ Document ready for questions!")

            # Question input
            st.markdown("### 💬 Ask About Your Document")
            question = st.text_input(
                "Type your question here and press Enter:",
                placeholder="What is the main idea of this document?",
                key="question_input"
            )

            if question.strip():
                with st.spinner("🤖 Analyzing document and generating answer..."):
                    answer = get_answer(question, store)
                
                st.markdown("### 📝 Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
else:
    st.info("👋 Welcome! Please upload a PDF document to get started.")
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=150)