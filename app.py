import os
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import deeplake

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
deeplake_account_name = os.getenv("DEEPLAKE_ACCOUNT_NAME")

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="LLM Powered Document Chatbot",
    page_icon="🤖",
    layout="centered",
)

def get_embedding_function(api_key):
    """Returns a function that embeds a list of texts using OpenAI API."""
    client = OpenAI(api_key=api_key)
    def embedding_function(texts):
        if isinstance(texts, str):
            texts = [texts]
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    return embedding_function

def doc_preprocessing():
    """Reads PDF files from data/ directory and returns list of text chunks with metadata."""
    if not os.path.exists('data'):
        os.makedirs('data')
        
    chunks = []
    metadata = []
    
    # List all PDF files
    pdf_files = [f for f in os.listdir('data') if f.endswith('.pdf')]
    if not pdf_files:
        raise ValueError("No PDF documents found in the 'data/' directory. Please upload some PDFs.")
        
    for filename in pdf_files:
        filepath = os.path.join('data', filename)
        reader = PdfReader(filepath)
        text = ""
        for i, page in enumerate(reader.pages):
            text_content = page.extract_text()
            if text_content:
                text += text_content + "\n"
        
        # Simple character-based splitter (chunk size 1000 characters, overlap 150)
        chunk_size = 1000
        chunk_overlap = 150
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"source": filename, "chunk_index": len(chunks) - 1})
            if end == len(text):
                break
            start += chunk_size - chunk_overlap
            
    return chunks, metadata

@st.cache_resource
def get_vector_store(api_key, token, account_name):
    """Loads or creates the Deep Lake vector store."""
    is_local = not (token and account_name)
    dataset_path = "./deeplake_db" if is_local else f"hub://{account_name}/text_embedding"
    
    embed_fn = get_embedding_function(api_key)
    
    # Check if database already exists
    db_exists = False
    if is_local:
        if os.path.exists(dataset_path) and os.listdir(dataset_path):
            db_exists = True
    else:
        try:
            # Check cloud database existence
            db = deeplake.VectorStore(path=dataset_path, read_only=True, token=token)
            if len(db) > 0:
                db_exists = True
        except Exception:
            db_exists = False

    if db_exists:
        st.write(f"ℹ️ Loading existing vector store from: `{dataset_path}`")
        db = deeplake.VectorStore(
            path=dataset_path,
            read_only=True,
            embedding_function=embed_fn,
            token=token if not is_local else None
        )
    else:
        st.write(f"🚀 Creating new vector store at: `{dataset_path}`...")
        chunks, metadata = doc_preprocessing()
        
        db = deeplake.VectorStore(
            path=dataset_path,
            overwrite=True,
            embedding_function=embed_fn,
            token=token if not is_local else None
        )
        
        # Add to vector store
        db.add(
            text=chunks,
            metadata=metadata
        )
        
        # Reload as read-only to avoid concurrency conflicts
        db = deeplake.VectorStore(
            path=dataset_path,
            read_only=True,
            embedding_function=embed_fn,
            token=token if not is_local else None
        )
    return db

def ask_question(db, api_key, query):
    """Queries the database and uses OpenAI to answer the question based on retrieved contexts."""
    # Retrieve top chunks (k=6)
    results = db.search(query=query, k=6)
    retrieved_texts = results.get("text", [])
    
    # Format context
    context = "\n\n---\n\n".join(retrieved_texts)
    
    # Build prompt
    system_instruction = (
        "You are an assistant designed to answer questions about the provided documents.\n"
        "Use the retrieved text snippets below to answer the user's question. "
        "If the answer cannot be found in the snippets, say that you don't know based on the documents.\n\n"
        f"--- DOCUMENT CONTENT ---\n{context}\n----------------------"
    )
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def main():
    st.title("🤖 LLM Powered Document Chatbot")
    st.write(
        "Ask questions about your PDF documents. The app uses OpenAI and Deep Lake "
        "to retrieve relevant context and answer your questions."
    )
    
    # Check if OpenAI API Key is available
    if not openai_api_key:
        st.error(
            "🔑 **OpenAI API Key is missing!** Please create a `.env` file in the project folder "
            "and set your key as `OPENAI_API_KEY=your_key` or set it in your environment."
        )
        st.info("You can use `.env.template` as a starting point. Copy it to `.env` and fill in the values.")
        return

    # Display storage mode info
    is_local = not (activeloop_token and deeplake_account_name)
    if is_local:
        st.info("💡 **Running in Local Mode:** Deep Lake will save the vector database locally in the `./deeplake_db` folder.")
    else:
        st.success(f"🌐 **Running in Cloud Mode:** Deep Lake will connect to `hub://{deeplake_account_name}/text_embedding`.")

    # Initialize vector database
    try:
        db = get_vector_store(openai_api_key, activeloop_token, deeplake_account_name)
    except Exception as e:
        st.error(f"Failed to initialize the vector database: {e}")
        st.info("Please make sure you have PDF files in the `data/` directory and your API keys are correct.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! I am ready to help you with your documents."}
        ]

    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Accept user input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_question(db, openai_api_key, user_input)
                    st.markdown(response)
                except Exception as e:
                    response = f"⚠️ An error occurred while fetching the answer: {e}"
                    st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
