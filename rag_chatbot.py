"""
Simple RAG Chatbot (teaching version)
=====================================

RAG = Retrieval Augmented Generation.

The idea in one line:
    Before asking the LLM a question, FIND the relevant text from your own
    documents and paste it into the prompt. The LLM then answers from that text
    instead of guessing from its training data.

Pipeline (each step below matches a numbered section in this file):

    DOCUMENTS --> [1 load] --> [2 chunk] --> [3 embed] --> in-memory index
                                                               |
    QUESTION  --> [4 embed question] --> [5 retrieve top-k] <--+
                                               |
                                   [6 build prompt] --> [7 LLM] --> ANSWER

Run it:
    python rag_chatbot.py              # normal RAG mode
    python rag_chatbot.py --no-rag     # plain LLM, no retrieval (for comparison)
    python rag_chatbot.py --quiet      # hide the retrieved chunks

We deliberately do NOT use a vector database here. A numpy array is enough for
a few hundred chunks, and it lets students see exactly what a "vector store"
actually does (it stores vectors and finds the closest ones).
"""

import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Read the .env file so GROQ_API_KEY etc. are available via os.getenv().
load_dotenv()

# Windows consoles use an old text encoding and can crash on characters like "₹".
# This line makes printing always use UTF-8.
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# SETTINGS
# Change these while learning and see what happens.
# ---------------------------------------------------------------------------

# The folder with our documents (.txt and .md files). It sits next to this file.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Chunking: how we cut documents into pieces.
CHUNK_SIZE = 500       # characters per chunk. Too big = noisy, too small = no context.
CHUNK_OVERLAP = 100    # characters shared between neighbouring chunks, so a sentence
                       # cut in half at a boundary still appears whole in one chunk.
                       # Must be smaller than CHUNK_SIZE.

# Retrieval: how many chunks we hand to the LLM.
TOP_K = 3

# Embedding model: turns text into a vector (list of numbers) that captures meaning.
# all-MiniLM-L6-v2 is small (~90 MB), fast on CPU, and downloads automatically.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM: Groq offers an OpenAI compatible API, so we use the normal OpenAI client
# and just point it at Groq's URL. To use real OpenAI or a local model (Ollama),
# change LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env. No code change needed.
# os.getenv("NAME", default) reads a value from .env, or uses the default.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    LLM_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# 1. LOAD DOCUMENTS
# ---------------------------------------------------------------------------

def load_documents(folder):
    """Read every .txt and .md file in the folder.

    Returns a list like:  [{"source": "menu.txt", "text": "...full text..."}, ...]
    We keep the file name so we can show WHERE a piece of text came from.
    """
    docs = []
    for file_name in sorted(os.listdir(folder)):
        name = file_name.lower()
        if name.endswith(".txt") or name.endswith(".md"):
            file_path = os.path.join(folder, file_name)
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            docs.append({"source": file_name, "text": text})
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Slide a window over the text and cut it into overlapping pieces.

    Example with size=10, overlap=3 on "ABCDEFGHIJKLMNOP":
        chunk 1: ABCDEFGHIJ
        chunk 2:        HIJKLMNOP   (starts 7 characters later, repeats HIJ)

    Why chunk at all? Two reasons:
      a) An embedding of a whole book is a blurry average of everything.
         A small chunk has one clear meaning, so search is sharper.
      b) The LLM prompt has a size limit, and shorter prompts are cheaper.

    This is the simplest possible chunker (fixed characters). Real projects
    often split on paragraphs or sentences instead. Good exercise for students.
    """
    if overlap >= size:
        # Otherwise the window would never move forward.
        raise ValueError("CHUNK_SIZE must be bigger than CHUNK_OVERLAP")

    chunks = []
    step = size - overlap      # how far the window moves each time
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece != "":        # skip empty pieces at the end
            chunks.append(piece)
        start = start + step
    return chunks


def build_chunks(docs):
    """Chunk every document and remember which file each chunk came from."""
    all_chunks = []
    for doc in docs:
        for piece in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": piece})
    return all_chunks


# ---------------------------------------------------------------------------
# 3. EMBED  (build the "index")
# ---------------------------------------------------------------------------

def build_index(chunks, embedder):
    """Turn every chunk into a vector.

    An embedding is a list of numbers (384 of them for this model).
    Texts with similar MEANING end up with similar numbers, even if they share
    no words. "refund" and "get my money back" land close together.

    normalize_embeddings=True scales each vector to length 1. That makes the
    similarity maths simple later (see retrieve()).

    Result: a numpy array with one row per chunk and 384 columns,
    shape (number_of_chunks, 384). This array IS our vector store.
    """
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# 4 + 5. EMBED THE QUESTION, THEN RETRIEVE
# ---------------------------------------------------------------------------

def get_score(result):
    """Used by sort() below: sort the results by their score."""
    return result["score"]


def retrieve(question, chunks, index, embedder, k=TOP_K):
    """Find the k chunks whose meaning is closest to the question."""

    # 4. Embed the question with the SAME model used for the chunks.
    #    Using a different model would put vectors in a different "space" and
    #    comparisons would be meaningless.
    question_vector = embedder.encode(question, normalize_embeddings=True)

    # 5. Give every chunk a score with cosine similarity: how much two vectors
    #    point the same way. 1.0 = very similar, 0 = unrelated, negative = opposite.
    #    Because every vector has length 1, cosine similarity is just a dot product.
    results = []
    for i, chunk in enumerate(chunks):
        chunk_vector = index[i]
        score = float(np.dot(chunk_vector, question_vector))
        results.append({"source": chunk["source"], "text": chunk["text"], "score": score})

    # Highest score first, then keep only the best k.
    results.sort(key=get_score, reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# 6. BUILD THE PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided. If the answer is not in the context, say you don't have that "
    "information. Do not make anything up. Keep answers short and clear."
)
# That "ONLY" and the "say you don't know" line matter a lot. Without them the
# model happily mixes in its own knowledge and the whole point of RAG is lost.


def build_messages(question, retrieved):
    """Put the retrieved chunks into the prompt, above the question."""
    context = ""
    for r in retrieved:
        context = context + f"[Source: {r['source']}]\n{r['text']}\n\n"

    user_prompt = f"Context:\n{context}Question: {question}"

    # A chat model takes a list of messages. The "system" message sets the rules,
    # the "user" message is what we are asking.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages


# ---------------------------------------------------------------------------
# 7. CALL THE LLM
# ---------------------------------------------------------------------------

def ask_llm(client, messages):
    """Send the messages to the model and return its text answer."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,   # 0 = least random. For factual Q&A we want stable answers.
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# MAIN CHAT LOOP
# ---------------------------------------------------------------------------

def main():
    # Read the command line options, e.g.  python rag_chatbot.py --quiet
    use_rag = "--no-rag" not in sys.argv
    show_chunks = "--quiet" not in sys.argv

    if not LLM_API_KEY or LLM_API_KEY.startswith("your_"):
        sys.exit("No API key found. Copy .env.example to .env and add your key.")

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    # Steps 1 to 3 happen ONCE at startup, not on every question.
    if use_rag:
        print("Loading documents and building the index...")
        docs = load_documents(DATA_DIR)
        if len(docs) == 0:
            sys.exit(f"No .txt or .md files found in {DATA_DIR}")

        chunks = build_chunks(docs)
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        index = build_index(chunks, embedder)
        print(f"Ready. {len(docs)} files -> {len(chunks)} chunks "
              f"-> index shape {index.shape}\n")
        print("Mode: RAG. Type 'exit' to quit.\n")
    else:
        print("Mode: plain LLM (no RAG). Type 'exit' to quit.\n")

    # Steps 4 to 7 happen for EVERY question.
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", ""]:
            break

        if use_rag:
            retrieved = retrieve(question, chunks, index, embedder)

            if show_chunks:
                print("\n--- retrieved chunks ---")
                for r in retrieved:
                    preview = r["text"].replace("\n", " ")[:90]
                    print(f"  {r['score']:.2f}  {r['source']}: {preview}...")
                print("------------------------")

            messages = build_messages(question, retrieved)
        else:
            # No retrieval: just send the question on its own.
            messages = [{"role": "user", "content": question}]

        answer = ask_llm(client, messages)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
