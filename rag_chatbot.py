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

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Read the .env file so GROQ_API_KEY etc. are available via os.getenv().
load_dotenv()

# Windows consoles default to a legacy encoding and crash on characters like "₹".
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# SETTINGS
# Change these while teaching and let students see what happens.
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"   # folder containing .txt / .md files

# Chunking: how we cut documents into pieces.
CHUNK_SIZE = 500       # characters per chunk. Too big = noisy, too small = no context.
CHUNK_OVERLAP = 100    # characters shared between neighbouring chunks, so a sentence
                       # cut in half at a boundary still appears whole in one chunk.

# Retrieval: how many chunks we hand to the LLM.
TOP_K = 3

# Embedding model: turns text into a vector (list of numbers) that captures meaning.
# all-MiniLM-L6-v2 is small (~90 MB), fast on CPU, and downloads automatically.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM: Groq exposes an OpenAI compatible API, so we use the normal OpenAI client
# and just point it at Groq's URL. To use real OpenAI or a local model (Ollama),
# change LLM_BASE_URL and LLM_API_KEY in .env. No code change needed.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# 1. LOAD DOCUMENTS
# ---------------------------------------------------------------------------

def load_documents(folder: Path) -> list[dict]:
    """Read every .txt and .md file in `folder`.

    Returns a list like:  [{"source": "menu.txt", "text": "...full text..."}, ...]
    We keep the file name so we can show the user WHERE an answer came from.
    """
    docs = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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
    chunks = []
    step = size - overlap          # how far the window moves each time
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:                  # skip empty pieces at the end
            chunks.append(piece)
    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    """Chunk every document and remember which file each chunk came from."""
    all_chunks = []
    for doc in docs:
        for piece in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": piece})
    return all_chunks


# ---------------------------------------------------------------------------
# 3. EMBED  (build the "index")
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict], embedder: SentenceTransformer) -> np.ndarray:
    """Turn every chunk into a vector.

    An embedding is a list of numbers (384 of them for this model).
    Texts with similar MEANING end up with similar numbers, even if they share
    no words. "refund" and "get my money back" land close together.

    normalize_embeddings=True scales each vector to length 1. That makes the
    similarity maths trivial later (see retrieve()).

    Result shape: (number_of_chunks, 384). This array IS our vector store.
    """
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# 4 + 5. EMBED THE QUESTION, THEN RETRIEVE
# ---------------------------------------------------------------------------

def retrieve(question: str, chunks: list[dict], index: np.ndarray,
             embedder: SentenceTransformer, k: int = TOP_K) -> list[dict]:
    """Find the k chunks whose meaning is closest to the question."""

    # 4. Embed the question with the SAME model used for the chunks.
    #    Using a different model would put vectors in a different "space" and
    #    comparisons would be meaningless.
    q_vec = embedder.encode(question, normalize_embeddings=True)

    # 5. Cosine similarity = how much two vectors point the same way.
    #    1.0 = same direction (very similar), 0 = unrelated, negative = opposite.
    #    Because every vector has length 1, cosine similarity is just a dot product.
    #    index @ q_vec scores ALL chunks in one go.
    scores = index @ q_vec

    # argsort gives indices from lowest to highest score. We flip it and take k.
    top_ids = np.argsort(scores)[::-1][:k]

    return [{**chunks[i], "score": float(scores[i])} for i in top_ids]


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


def build_messages(question: str, retrieved: list[dict]) -> list[dict]:
    """Stuff the retrieved chunks into the prompt, above the question."""
    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# 7. CALL THE LLM
# ---------------------------------------------------------------------------

def ask_llm(client: OpenAI, messages: list[dict]) -> str:
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG chatbot")
    parser.add_argument("--no-rag", action="store_true",
                        help="skip retrieval and ask the LLM directly (for comparison)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the retrieved chunks")
    args = parser.parse_args()

    if not LLM_API_KEY or LLM_API_KEY.startswith("your_"):
        sys.exit("No API key found. Copy .env.example to .env and add your key.")

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    # Index building happens ONCE at startup, not on every question.
    chunks, index, embedder = [], None, None
    if not args.no_rag:
        print("Loading documents and building the index...")
        docs = load_documents(DATA_DIR)
        if not docs:
            sys.exit(f"No .txt or .md files found in {DATA_DIR}")
        chunks = build_chunks(docs)
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        index = build_index(chunks, embedder)
        print(f"Ready. {len(docs)} files -> {len(chunks)} chunks "
              f"-> index shape {index.shape}\n")

    mode = "plain LLM (no RAG)" if args.no_rag else "RAG"
    print(f"Mode: {mode}. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break

        if args.no_rag:
            messages = [{"role": "user", "content": question}]
        else:
            retrieved = retrieve(question, chunks, index, embedder)
            if not args.quiet:
                print("\n--- retrieved chunks ---")
                for r in retrieved:
                    preview = r["text"].replace("\n", " ")[:90]
                    print(f"  {r['score']:.2f}  {r['source']}: {preview}...")
                print("------------------------")
            messages = build_messages(question, retrieved)

        print(f"\nBot: {ask_llm(client, messages)}\n")


if __name__ == "__main__":
    main()
