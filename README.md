# Simple RAG Chatbot (teaching project)

A small chatbot that answers questions from your own documents. One Python file, no vector database, no framework. The goal is that students can read the whole thing in ten minutes and see exactly what RAG does.

The demo data is a made up cafe (Marina Bean Cafe) with a menu and a policy sheet. The LLM has never heard of this cafe, so any correct answer has to come from retrieval. That makes the effect of RAG obvious.

---

## 1. Quick start

Needs Python 3.9 or newer.

```bash
git clone https://github.com/saravanagithub14/Rag-teaching-.git
cd Rag-teaching-
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env             # macOS/Linux: cp .env.example .env
# then open .env and replace your_groq_key_here with your Groq key
python rag_chatbot.py
```

Get a free Groq key at https://console.groq.com. First run downloads the embedding model (about 90 MB), so do this before class, not during. A warning about `HF_TOKEN` on startup is harmless; it just means the model is downloaded without a Hugging Face login.

The chatbot runs in the terminal. Type a question at the `You:` prompt, and type `exit` (or press Enter on an empty line) to quit.

Flags:

| Command | What it does |
|---|---|
| `python rag_chatbot.py` | Normal RAG mode, shows retrieved chunks |
| `python rag_chatbot.py --no-rag` | Plain LLM, no retrieval. Use for the "before" demo |
| `python rag_chatbot.py --quiet` | RAG mode, hides the retrieved chunks |

### Using a different LLM

The code reads these optional values from `.env`. Leave them out to use Groq.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (none) | Your Groq key. Used when `LLM_API_KEY` is not set |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI compatible API, e.g. OpenAI or a local Ollama server |
| `LLM_API_KEY` | (none) | Key for that API. Takes priority over `GROQ_API_KEY` |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name on that API |

For a local Ollama model no key is needed, but `LLM_API_KEY` can't be empty, so put any placeholder text in it. Commented examples for OpenAI and Ollama are in `.env.example`.

---

## 2. What is RAG, in plain words

LLMs know a lot, but they don't know **your** stuff: your company policies, your syllabus, yesterday's prices. And when they don't know, they often guess confidently. That's a hallucination.

RAG fixes it in two moves:

1. **Retrieve**: search your own documents for the pieces relevant to the question.
2. **Generate**: paste those pieces into the prompt and tell the LLM to answer from them.

The model isn't retrained. It just gets an open book exam instead of a closed book one.

---

## 3. The pipeline

```
DOCUMENTS --> [1 load] --> [2 chunk] --> [3 embed] --> in-memory index
                                                           |
QUESTION  --> [4 embed question] --> [5 retrieve top-k] <--+
                                           |
                               [6 build prompt] --> [7 LLM] --> ANSWER
```

Steps 1 to 3 run **once** at startup (the "indexing" phase). Steps 4 to 7 run on **every question** (the "query" phase). Worth saying out loud in class, because students often think the whole thing reruns each time.

Each step has a matching numbered section in `rag_chatbot.py`.

### Step 1: Load
Read every `.txt` and `.md` file from `data/`. We keep the file name with the text so the retrieved chunks (and the prompt) can show where each piece came from.

### Step 2: Chunk
Cut each document into 500 character pieces, with 100 characters of overlap.

- **Why cut at all?** An embedding of a whole document is a blurry average of everything in it. A small chunk has one clear meaning, so search gets sharper. Also, prompts have size limits and cost money per token.
- **Why overlap?** Without it, a sentence cut in half at a boundary is lost to both chunks. With overlap, it shows up whole in at least one.
- **Limitation:** we cut by character count, so we sometimes cut mid word. Real projects split on paragraphs or sentences. Good student exercise.
- **With the demo data:** the two files become just 6 chunks, so the startup line reads `2 files -> 6 chunks -> index shape (6, 384)`.

### Step 3: Embed
An embedding model turns text into a list of numbers (384 of them for `all-MiniLM-L6-v2`). Texts with similar **meaning** get similar numbers, even with no shared words. "Refund" and "get my money back" land close together. That is the whole trick behind semantic search.

All chunk vectors stacked together make a numpy array of shape `(number_of_chunks, 384)`. That array **is** our vector store. A vector database is basically this array plus fancy indexing for when you have millions of rows.

### Step 4: Embed the question
Same model, same space. Embedding the question with a different model would make the comparison meaningless, like measuring one thing in cm and another in inches.

### Step 5: Retrieve
Score every chunk against the question using **cosine similarity** (how much two vectors point the same way: 1 means same direction, 0 unrelated). We normalised every vector to length 1 earlier, so cosine similarity is just a dot product: `index @ q_vec`. Sort, take the top `k` (default 3).

### Step 6: Build the prompt
Put the retrieved chunks above the question, labelled with their source file. The system prompt says "use ONLY the context" and "say you don't know if it's not there". Those two instructions do a lot of work. Remove them in class and watch the model start mixing in its own guesses.

### Step 7: Call the LLM
Groq offers an OpenAI compatible API, so we use the normal `openai` library and point it at Groq's URL. `temperature=0` keeps factual answers stable.

---

## 4. The settings worth playing with

The first four are in the SETTINGS block at the top of `rag_chatbot.py`. `LLM_MODEL` is easiest to change in `.env`. `SYSTEM_PROMPT` is in section 6 of the file.

| Setting | Default | Try this |
|---|---|---|
| `CHUNK_SIZE` | 500 | Set to 100, then 2000. Ask the same question each time |
| `CHUNK_OVERLAP` | 100 | Set to 0 and ask about something near a chunk boundary |
| `TOP_K` | 3 | Set to 1 (misses answers spread over chunks), then 10 (noise) |
| `EMBED_MODEL_NAME` | all-MiniLM-L6-v2 | Try another sentence-transformers model and compare scores |
| `LLM_MODEL` | openai/gpt-oss-20b | Try a bigger model and compare |
| `SYSTEM_PROMPT` | "ONLY the context..." | Delete the "ONLY" and see what happens |

---

## 5. Classroom demo script (about 30 minutes)

**Part A: the problem (5 min).** Run with `--no-rag`.

- "What is the price of the Coconut Cold Brew at Marina Bean Cafe?"
- "What time does Marina Bean Cafe close?"

The model will either say it doesn't know or invent something. Both are useful to show. Ask students which they'd trust.

**Part B: the fix (5 min).** Run normally, ask the same two questions. Point at the retrieved chunks and their scores. The right chunk should be among the top 3, usually first. Then the answer is correct and short. (With this tiny dataset scores are modest, often 0.3 to 0.5, and the right chunk is sometimes second. That's a good moment to explain why `TOP_K` is more than 1.)

**Part C: peek inside (10 min).** Open `rag_chatbot.py`, walk the numbered sections top to bottom. Spend the most time on step 3 and 5, that's where the idea lives.

**Part D: break it (10 min).** Let students try:

- A question the documents don't answer ("Do you sell pizza?"). Good result: "I don't have that information".
- A vague question ("tell me about food"). Watch the scores drop.
- A question using different words from the document ("can I bring my dog?" vs "pets"). This shows embeddings matching meaning, not keywords.
- A question needing two facts that sit in different chunks ("I want a Coconut Cold Brew with oat milk, delivered. What's the total?"). This needs `TOP_K` high enough to pull all the chunks involved.

### Questions with known answers (for checking)

| Question | Answer | File |
|---|---|---|
| What is the signature drink and its price? | Coconut Cold Brew, 160 rupees | menu_and_prices.txt |
| How much extra for oat milk? | 30 rupees | menu_and_prices.txt |
| When is the appam and stew special? | Tuesdays until 11 am, 220 rupees | menu_and_prices.txt |
| When does the cafe close? | 10 pm daily, 2 pm on the first Monday | about_and_policies.txt |
| Is there free delivery? | Yes, on orders above 600 rupees | about_and_policies.txt |
| Can I bring my dog? | Yes, on the terrace only | about_and_policies.txt |
| How long do I have to complain about a wrong order? | 15 minutes | about_and_policies.txt |
| Who founded the cafe? | Anjali Nair and her brother Rohit | menu_and_prices.txt |

---

## 6. Common problems and what they teach

**The right chunk wasn't retrieved.** Look at the scores. Usually the chunk is too big (diluted meaning) or too small (cut away from its context). Change `CHUNK_SIZE`.

**The right chunk was retrieved but the answer is wrong.** That's a generation problem, not a retrieval problem. Check the prompt. Debugging RAG means always asking: did retrieval fail, or did the LLM fail? Different fixes.

**The bot answers things not in the documents.** The system prompt is too loose, or the model is too small to follow it. Tighten the prompt.

**Scores all look low (below 0.3) but the bot still answers.** Nothing stops it right now. See the exercises, add a minimum score threshold.

**"No API key found" on startup.** `.env` is missing, or `GROQ_API_KEY` still holds the `your_groq_key_here` placeholder. Copy `.env.example` to `.env` and paste your real key.

---

## 7. Limits of this version (be honest with students)

- **Index rebuilds on every start.** Fine for a few files. For more, save the array with `np.save` and load it.
- **Brute force search.** We compare against every chunk. Perfect for hundreds of chunks, slow for millions. That's when you'd bring in FAISS, Chroma or pgvector.
- **No memory.** Each question stands alone. "And how much is that one?" won't work, because the bot doesn't remember "that one".
- **Only `.txt` and `.md`.** PDFs and Word files need a text extraction step first.
- **Fixed size chunking.** Cuts mid sentence.
- **No reranking.** Real systems often retrieve 20 chunks and use a second model to pick the best 3.

---

## 8. Exercises for students

Roughly easiest to hardest.

1. Add your own document to `data/` (class timetable, a college handbook) and ask questions about it.
2. Change `TOP_K` and `CHUNK_SIZE`, and write down what changes.
3. Add a **minimum score threshold**: if the best score is below, say, 0.3, skip the LLM and reply "I don't have that information". This saves tokens and cuts hallucinations.
4. Show the **source file names** at the end of each answer.
5. Change `chunk_text` to split on paragraphs (blank lines) instead of characters.
6. Add **chat memory**: keep the last few messages and send them with each request.
7. Save and load the index with `np.save` / `np.load` so startup is instant.
8. Add PDF support using `pypdf`.
9. Swap the numpy search for ChromaDB and compare.

---

## 9. Vocabulary cheat sheet

| Term | Meaning |
|---|---|
| **RAG** | Retrieval Augmented Generation. Look things up, then answer |
| **Chunk** | A small piece of a document, the unit we search over |
| **Embedding** | A list of numbers representing the meaning of a text |
| **Vector store** | Where embeddings live. Here, a numpy array |
| **Cosine similarity** | Score for how close two meanings are, -1 to 1 |
| **Top-k** | The k best matching chunks we hand to the LLM |
| **Context** | The retrieved text pasted into the prompt |
| **Hallucination** | The model confidently saying something false |
| **System prompt** | Standing instructions telling the model how to behave |
| **Temperature** | Randomness of output. 0 is steady, higher is more creative |

---

## 10. Project layout

```
Rag-teaching-/
  rag_chatbot.py        the whole chatbot, annotated
  data/
    menu_and_prices.txt
    about_and_policies.txt
  requirements.txt      numpy, python-dotenv, openai, sentence-transformers
  .env.example          copy to .env and add your key
  .env                  your real key (you create it; git-ignored, never commit it)
  .gitignore            standard Python ignores, including .env and .venv/
  README.md             this file
```

Swap anything in `data/` for your own `.txt` or `.md` files and the bot works on those instead. No code change.
