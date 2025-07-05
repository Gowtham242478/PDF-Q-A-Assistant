from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_text(text):
    """Split PDF text into optimized chunks with logging"""
    logger.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def create_vector_store(chunks):
    """Create FAISS vector store with progress tracking"""
    if not chunks:
        raise ValueError("No content found in PDF to index.")
    
    logger.info("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = FAISS.from_texts(chunks, embeddings)
    logger.info("Vector store created successfully")
    return store

def get_answer(query, vector_store):
    """Generate answer with enhanced prompt engineering"""
    logger.info(f"Processing question: {query}")
    
    # Retrieve relevant context
    docs = vector_store.similarity_search(query, k=1)
    context = "\n".join([doc.page_content for doc in docs])
    logger.info(f"Context length: {len(context)} characters")

    # Load model and tokenizer
    model_name = "google/flan-t5-base"
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Create pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )

    # Enhanced prompt template
    prompt = f"""
    Please provide a clear and concise answer to the question based on the given context.
    If the context doesn't contain enough information, respond with "I couldn't find enough information to answer that question."

    Context: {context}

    Question: {query}

    Answer:
    """
    
    logger.info(f"Prompt: {prompt[:200]}...")  # Log truncated prompt
    result = pipe(prompt)[0]['generated_text']
    logger.info(f"Generated answer: {result}")
    
    return result