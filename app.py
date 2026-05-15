import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="The Hogwarts Archivist", page_icon="🪄", layout="centered")

# Custom CSS for a magical feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #f0e6d2;
    }
    .stButton>button {
        background-color: #741b47;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🪄 The Hogwarts Archivist")
st.markdown("---")
st.write("*Welcome, seeker of knowledge. Ask me anything from the seven books, and I shall search the Restricted Section for you.*")

# --- INITIALIZATION & SECRETS ---
# Accessing the Groq API Key from Hugging Face Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Error: GROQ_API_KEY not found in Hugging Face Secrets. Please add it in Settings > Variables and Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- LOAD VECTOR DATABASE (CACHED) ---
@st.cache_resource
def load_vector_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Ensure the 'faiss_hp_index' folder is uploaded to your Space
        vector_db = FAISS.load_local("faiss_hp_index", embeddings, allow_dangerous_deserialization=True)
        return vector_db
    except Exception as e:
        st.error(f"Failed to load the Magical Index: {e}")
        return None

vector_db = load_vector_db()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to know?"):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG Logic: Retrieval
    with st.spinner("Searching the library stacks..."):
        # Search the top 3 relevant chunks
        related_docs = vector_db.similarity_search(prompt, k=3)
        context_text = "\n\n".join([doc.page_content for doc in related_docs])
        sources = [doc.metadata.get("source", "Unknown Tome") for doc in related_docs]

        # 3. RAG Logic: Generation
        system_prompt = f"""
        You are the 'Hogwarts Archivist', a wise and helpful magical librarian. 
        You only answer questions based on the following book context:
        {context_text}
        
        If the answer is not in the context, say: "Alas, that lore is hidden even from the Restricted Section."
        Always maintain a magical, polite, and scholarly tone.
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-8b-instant", # Updated model name
            )
            response = chat_completion.choices[0].message.content
            
            # 4. Display response
            with st.chat_message("assistant"):
                st.markdown(response)
                # Show sources in an expander
                with st.expander("📚 View Lore Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i+1}:** {source}")
                        st.info(related_docs[i].page_content)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"The magic fizzled out: {e}")
