# 🧙‍♂️ The Hogwarts Archivist - RAG Chatbot

The Hogwarts Archivist is a Retrieval-Augmented Generation (RAG) system designed to answer questions exclusively based on the Harry Potter book series. It acts as an enchanted librarian, providing highly accurate, source-cited answers while strictly refusing to hallucinate or answer real-world questions.

This project was built to demonstrate full-stack AI engineering, moving from data ingestion and vectorization in Google Colab to a production-ready Streamlit deployment on Hugging Face Spaces.

---

## ✨ Key Features

- **Accurate Lore Retrieval:** Uses semantic search to read through 95,000+ chunks of book text and retrieve the exact paragraphs containing the answer.
- **Source Attribution:** Proves its faithfulness by providing an expander UI that displays the raw book text, book title, and chapter it used to generate the answer.
- **Strict Context Enforcement (Guardrails):** Employs prompt engineering to prevent hallucinations. If asked about real-world events (e.g., "What is a combustion engine?" or "Who is the President?"), it gracefully responds: "Alas, that lore is hidden even from the Restricted Section."
- **Lightning-Fast Inference:** Uses the Groq API (llama-3.1-8b-instant) to generate magical, themed responses in milliseconds.

---

## 🏗️ Architecture & Workflow

The project was built in three distinct phases to optimize compute and deployment costs:

### Phase 1: Data Ingestion & Pre-processing (Google Colab)
- **Dataset:** Loaded the complete text of all 7 Harry Potter books from a public GitHub CSV ([gastonstat/harry-potter-data](https://github.com/gastonstat/harry-potter-data)).
- **Chunking Strategy:** Used LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`. The overlap ensures that magical secrets or dialogues split across chunks retain their context, resulting in 95,085 overlapping chunks.
- **Vectorization:** Converted the text into embeddings using the fully open-source, lightweight all-MiniLM-L6-v2 Hugging Face model.
- **Vector Storage:** Compiled the embeddings into a local FAISS index, exported as a `.zip` file to avoid recalculating embeddings during deployment.

### Phase 2: Core Logic & Prompt Engineering
- **Top-K Retrieval:** Configured FAISS to pull the top 5 most relevant documents (`k=5`) to prevent "Retrieval Sparsity" (e.g., ensuring minor characters like Neville Longbottom aren't missed during major plot queries).
- **The "Brain":** Integrated Groq API for rapid LLM inference, guided by a strict System Prompt to maintain a magical persona and adhere only to the provided context.

### Phase 3: Deployment (Hugging Face Spaces)
- Built a Streamlit frontend (`app.py`) for a clean, user-friendly interface.
- Uploaded the pre-calculated FAISS index folder directly to Hugging Face via Git LFS, ensuring the app loads instantly without heavy startup processing.

---

## 📊 Accuracy & Evaluation Scorecard

To ensure production readiness, the Archivist was put through a "Human-in-the-Loop" Verification suite consisting of 30 diverse test questions.

| Test Category       | Total Questions | Success Rate | Engineering Notes                                                                              |
|---------------------|-----------------|--------------|-----------------------------------------------------------------------------------------------|
| Factual Retrieval   | 10              | 100%         | Flawlessly identified wands, houses, and core lore (e.g., Unforgivable Curses).               |
| Specific Details    | 10              | 80%          | High accuracy. Resolved early context fragmentation issues by adjusting retrieval from Top-3 to Top-5 chunks. |
| Negative Testing    | 10              | 100%         | Perfect guardrail adherence. Correctly rejected all non-HP queries, preventing hallucinations. |

---

## 🚀 How to Run Locally

If you'd like to run The Hogwarts Archivist on your local machine:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/hogwarts-archivist-rag.git
    cd hogwarts-archivist-rag
    ```

2. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Ensure you have `streamlit`, `langchain-community`, `faiss-cpu`, `sentence-transformers`, and `groq` installed).*

3. **Set your API Key:**
    Get a free API key from the Groq Console and set it as an environment variable:
    ```sh
    # Windows
    set GROQ_API_KEY="your_api_key_here"
    # Mac/Linux
    export GROQ_API_KEY="your_api_key_here"
    ```

4. **Run the Streamlit App:**
    ```sh
    streamlit run app.py
    ```

---

## 🛠️ Engineering Decisions (For Interviewers)

- **Why RAG over Fine-Tuning?**  
  I chose RAG because it provides Source Attribution (preventing the model from confidently hallucinating) and makes updating the knowledge base trivial and cost-effective. Fine-tuning is better for adjusting tone, but RAG is superior for knowledge-heavy fact retrieval.

- **Why FAISS & Pre-calculation?**  
  By using Google Colab's GPU to process the 95k text chunks into embeddings and saving the FAISS index as a static folder, I saved immense compute costs and startup time for the Hugging Face deployment.

---

Mischief Managed. ⚡
