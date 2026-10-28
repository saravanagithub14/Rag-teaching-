# RAG Chatbot: Student Practice Guide

In this practice you will run a small chatbot that answers questions from its own documents, see why it works, and then change and break it on purpose. The whole chatbot is one Python file, `rag_chatbot.py`, with no vector database and no framework, so you can read every line.

The documents describe a made up cafe, **Marina Bean Cafe**. No LLM has ever heard of it, so whenever the bot answers correctly, you know the answer came from the documents and not from the model's memory.

**By the end you should be able to:**

- explain what RAG (Retrieval Augmented Generation) is and why it is used
- name the 7 steps of the RAG pipeline and find each one in the code
- read the similarity scores and tell whether retrieval worked
- explain how chunk size, overlap and top-k change the results
- decide whether a wrong answer was a retrieval problem or a generation problem

**Theory first?** [RAG_NOTES.md](RAG_NOTES.md) explains every concept used here, with worked examples and self-check questions.

**How to work:** keep a notes file or notebook open. Many tasks ask you to write down what you see. Change **one thing at a time**, and put it back before the next task.

---

## Part 0: Setup

You need Python 3.9 or newer and a free Groq API key from https://console.groq.com.

```bash
git clone https://github.com/saravanagithub14/Rag-teaching-.git
cd Rag-teaching-
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env             # macOS/Linux: cp .env.example .env
```

Open `.env`, replace `your_groq_key_here` with your key, and save. **Never commit `.env` to git**, because it holds your secret key. (It is already in `.gitignore`.)

Start the bot:

```bash
python rag_chatbot.py
```

The first run downloads the embedding model (about 90 MB), so it takes a minute. A warning about `HF_TOKEN` is harmless.

**Checkpoint.** You should see:

```
Ready. 2 files -> 6 chunks -> index shape (6, 384)

Mode: RAG. Type 'exit' to quit.

You:
```

Type a question at `You:`. To quit, type `exit` or press Enter on an empty line.

The program has three ways to run:

| Command | What it does |
|---|---|
| `python rag_chatbot.py` | RAG mode, prints the retrieved chunks and their scores |
| `python rag_chatbot.py --no-rag` | Plain LLM, no documents at all |
| `python rag_chatbot.py --quiet` | RAG mode, hides the retrieved chunks |

---

## Background: RAG in plain words

LLMs know a lot, but they don't know **your** information: a company's policies, your syllabus, today's prices. When they don't know, they often guess confidently. That is called a **hallucination**.

RAG fixes this in two moves:

1. **Retrieve:** search your own documents for the pieces that are relevant to the question.
2. **Generate:** paste those pieces into the prompt and tell the LLM to answer only from them.

The model is not retrained. It just gets an open book exam instead of a closed book one.

```
DOCUMENTS --> [1 load] --> [2 chunk] --> [3 embed] --> in-memory index
                                                           |
QUESTION  --> [4 embed question] --> [5 retrieve top-k] <--+
                                           |
                               [6 build prompt] --> [7 LLM] --> ANSWER
```

Steps 1 to 3 run **once**, when the program starts (indexing). Steps 4 to 7 run for **every question** (querying). Each step has a numbered section in `rag_chatbot.py`.

---

## Part 1: See the problem (no RAG)

Run the bot without retrieval:

```bash
python rag_chatbot.py --no-rag
```

Ask these questions and copy each answer into your notes:

1. What is the price of the Coconut Cold Brew at Marina Bean Cafe?
2. What time does Marina Bean Cafe close?
3. Who founded Marina Bean Cafe?

**Write down:** For each answer, did the model say it doesn't know, or did it invent something? Which of the two is more dangerous, and why?

---

## Part 2: Turn RAG on

Run the normal mode and ask the **same three questions**:

```bash
python rag_chatbot.py
```

Before each answer you will see the retrieved chunks, for example:

```
--- retrieved chunks ---
  0.45  about_and_policies.txt: lling the cafe.  Pets Well behaved dogs are welcome on the t...
  0.32  about_and_policies.txt: unds for food that has already been eaten. For delivery orde...
  0.04  about_and_policies.txt: counter for the password.  Ordering and payment We accept UP...
------------------------
```

Each line shows a **score** (how close that chunk's meaning is to your question; higher is closer), the **file** it came from, and the start of the chunk.

**Write down, for each question:**

| Question | Top score | Was the chunk with the answer in the list? At which position? | Was the answer correct? |
|---|---|---|---|
| | | | |

With this small dataset, scores of 0.3 to 0.5 are normal. The right chunk is sometimes second instead of first. **Think:** why does the program send 3 chunks to the LLM instead of only the best one?

Use the answer key at the end of this guide to check the bot.

---

## Part 3: Read the code

Open `rag_chatbot.py` and work through the numbered sections from top to bottom. Answer each question in your notes.

**Section 1, Load (`load_documents`)**
- Which file types does it read? What happens to a `.pdf` placed in `data/`?
- Why does it keep the file name next to the text?

**Section 2, Chunk (`chunk_text`)**
- With `CHUNK_SIZE = 500` and `CHUNK_OVERLAP = 100`, how many characters does the window move each step? (Look at `step` in the `while` loop.)
- Why is overlap useful? (Hint: what happens to a sentence that is cut in half at a chunk boundary?)
- Why not just embed each whole document as one piece?

**Section 3, Embed (`build_index`)**
- How many numbers does the model produce for one chunk? (Look at the startup line.)
- The result has shape `(6, 384)`. What does the 6 mean, and what does the 384 mean?
- In this program, what plays the role of the "vector store"?

**Sections 4 and 5, Retrieve (`retrieve`)**
- Why must the question be embedded with the **same** model as the chunks?
- What does `np.dot(chunk_vector, question_vector)` calculate, and why is a dot product enough here? (Hint: `normalize_embeddings=True`.)
- What do `results.sort(key=get_score, reverse=True)` and `results[:k]` do, step by step?

**Section 6, Build the prompt (`build_messages`, `SYSTEM_PROMPT`)**
- Where in the prompt do the chunks go, compared with the question?
- Which two instructions in `SYSTEM_PROMPT` stop the model from making things up?

**Section 7, Call the LLM (`ask_llm`)**
- The code uses the `openai` library, but talks to Groq. How? (Look at `LLM_BASE_URL`.)
- Why is `temperature=0` a good choice for answering factual questions?

---

## Part 4: Experiment with the settings

The settings are in the SETTINGS block at the top of `rag_chatbot.py` (`SYSTEM_PROMPT` is in section 6). For each experiment: change the value, save, restart the bot, ask the questions, write down what changed, then **put the value back**.

> **Rule:** `CHUNK_SIZE` must always be bigger than `CHUNK_OVERLAP`. Otherwise the window would never move forward, so the program stops with `ValueError: CHUNK_SIZE must be bigger than CHUNK_OVERLAP`.

**Experiment A: chunk size.** Use these questions each time: *"What is the signature drink and how much is it?"* and *"Can I bring my dog inside?"*

| `CHUNK_SIZE` | `CHUNK_OVERLAP` | Chunks at startup | Top score (Q1) | Top score (Q2) | Answers correct? |
|---|---|---|---|---|---|
| 150 | 30 | 19 | | | |
| 500 | 100 | 6 (default) | | | |
| 2000 | 100 | 2 | | | |

Check that your "chunks at startup" numbers match the table. **Think:** what happens to the meaning of a chunk when it is very big? When it is very small?

**Experiment B: overlap.** Set `CHUNK_OVERLAP = 0` (you should get 5 chunks). Look at where the chunks now start and end in the retrieved list. Do any of them cut a sentence in a way that loses the answer?

**Experiment C: top-k.** Ask the "total cost" question from the answer key with `TOP_K = 1`, then `3`, then `6`. The answer needs facts from three different chunks. Which setting gets it right? What is the downside of a very high `TOP_K` with a bigger document set?

**Experiment D: the system prompt.** Remove the word "ONLY" and the sentence about saying you don't have the information. Then ask *"Do you sell pizza?"* and *"What is the capital of France?"* What changes? Why does this matter for a real company chatbot?

**Optional.** Change `LLM_MODEL` in `.env` to another model available on Groq and compare answer quality.

> **Tip:** if you lose track of your changes, `git checkout -- rag_chatbot.py` puts the file back to the original.

---

## Part 5: Break it on purpose

With the default settings, try each of these and write down what happens:

1. **A question the documents don't answer:** *"Do you sell pizza?"* A good result is "I don't have that information". Look at the scores. Are they lower than usual?
2. **A vague question:** *"Tell me about food."* What happens to the scores? Is the answer still useful?
3. **Different words from the document:** *"Can I bring my puppy?"* The document says "Pets" and "dogs", not "puppy". Does retrieval still find the right chunk? This shows embeddings matching **meaning**, not exact words.
4. **A follow-up question:** ask *"How much is the cappuccino?"* then *"And with oat milk?"* Does the second answer work? Why not? (Hint: does the program remember earlier messages?)

---

## Part 6: Use your own documents

1. Create a `.txt` or `.md` file in `data/` about something you know well: your class timetable, a club's rules, a product manual.
2. Restart the bot. Check that the startup line shows 3 files and more chunks.
3. Write 5 questions whose answers are in your file, and 2 whose answers are not.
4. Record the score and correctness for each. Did retrieval mix up your file with the cafe files?

No code change is needed. The bot reads every `.txt` and `.md` file in `data/`.

---

## Part 7: Coding challenges

Roughly easiest to hardest. Each one says when you are done.

1. **Minimum score threshold.** If the best score is below 0.3, don't call the LLM; reply "I don't have that information".
   *Done when:* "Do you sell pizza?" gets that reply without an API call, and cafe questions still work.
   *Hint:* `retrieve()` already returns the scores, in order.

2. **Show sources.** After each answer, print the file names the answer came from, like `Sources: menu_and_prices.txt`.
   *Done when:* every answer ends with a sources line and no file is listed twice.

3. **Paragraph chunking.** Rewrite `chunk_text` to split on blank lines (`"\n\n"`) instead of a fixed number of characters.
   *Done when:* no chunk starts or ends in the middle of a word. Compare scores with the original.

4. **Chat memory.** Keep the last few questions and answers and send them with each new request.
   *Done when:* "And with oat milk?" works after a cappuccino question.

5. **Save the index.** Save the embeddings with `np.save` and load them with `np.load` on the next start, so startup is faster.
   *Done when:* the second start skips embedding. Think about when the saved file needs rebuilding.

6. **PDF support.** Use the `pypdf` library to read `.pdf` files in `data/`.

7. **Real vector database.** Replace the numpy search with ChromaDB. Compare the retrieved chunks with the original version.

---

## Debugging guide

When the bot gives a wrong answer, first ask: **did retrieval fail, or did the LLM fail?** They have different fixes.

| What you see | Probable cause | What to try |
|---|---|---|
| The chunk with the answer is not in the retrieved list | Retrieval failed | Change `CHUNK_SIZE`, raise `TOP_K`, rephrase the question |
| The right chunk is retrieved but the answer is wrong | Generation failed | Check the prompt; try a stronger `LLM_MODEL` |
| The bot answers things that are not in the documents | System prompt too loose, or model too small | Tighten `SYSTEM_PROMPT` |
| All scores are low but the bot still answers | Nothing stops it | Challenge 1, the score threshold |
| `No API key found` at startup | `.env` missing or still has the placeholder | Copy `.env.example` to `.env` and add your key |
| `ValueError: CHUNK_SIZE must be bigger than CHUNK_OVERLAP` | `CHUNK_OVERLAP` is equal to or bigger than `CHUNK_SIZE` | Make `CHUNK_SIZE` bigger than `CHUNK_OVERLAP` |
| `No .txt or .md files found` | `data/` is empty or has other file types | Add a `.txt` or `.md` file |

---

## Limits of this version

Knowing what a tool can't do is part of understanding it.

- **The index is rebuilt on every start.** Fine for a few files, slow for many (challenge 5).
- **Brute force search.** Every chunk is scored for every question. Fine for hundreds of chunks; for millions, you would use FAISS, Chroma or pgvector.
- **No memory.** Each question stands alone (challenge 4).
- **Only `.txt` and `.md` files** (challenge 6).
- **Fixed size chunking** cuts in the middle of words and sentences (challenge 3).
- **No reranking.** Real systems often retrieve 20 chunks and use a second model to choose the best 3.

---

## Answer key: cafe questions

Use this to check the bot's answers.

| Question | Correct answer | File |
|---|---|---|
| What is the signature drink and its price? | Coconut Cold Brew, 160 rupees | menu_and_prices.txt |
| How much extra for oat milk? | 30 rupees | menu_and_prices.txt |
| When is the appam and stew special? | Tuesdays until 11 am, 220 rupees | menu_and_prices.txt |
| When does the cafe close? | 10 pm daily; 2 pm on the first Monday of the month; closed on Onam | about_and_policies.txt |
| Is there free delivery? | Yes, on orders above 600 rupees (within 4 km) | about_and_policies.txt |
| Can I bring my dog? | Yes, on the terrace only, never inside | about_and_policies.txt |
| How long do I have to complain about a wrong order? | 15 minutes | about_and_policies.txt |
| Who founded the cafe? | Anjali Nair and her brother Rohit | menu_and_prices.txt |
| I want a Coconut Cold Brew with oat milk, delivered. What's the total? | 220 rupees (160 + 30 oat milk + 30 delivery, since the order is under 600) | both files |
| Do you sell pizza? | Not in the documents; the bot should say it doesn't have that information | none |

---

## Vocabulary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval Augmented Generation. Look things up, then answer |
| **Chunk** | A small piece of a document, the unit we search over |
| **Embedding** | A list of numbers representing the meaning of a text |
| **Vector store** | Where embeddings are kept. Here, a numpy array |
| **Cosine similarity** | Score for how close two meanings are, from -1 to 1 |
| **Top-k** | The k best matching chunks we give to the LLM |
| **Context** | The retrieved text pasted into the prompt |
| **Hallucination** | The model confidently saying something false |
| **System prompt** | Standing instructions telling the model how to behave |
| **Temperature** | How random the output is. 0 is steady, higher is more creative |

---

## Project files

```
Rag-teaching-/
  rag_chatbot.py        the whole chatbot, with comments
  data/
    menu_and_prices.txt
    about_and_policies.txt
  requirements.txt      numpy, python-dotenv, openai, sentence-transformers
  .env.example          copy to .env and add your key
  .env                  your real key (you create it; never commit it)
  .gitignore            standard Python ignores, including .env and .venv/
  README.md             this guide
  RAG_NOTES.md          theory notes on RAG
```

Optional `.env` settings, if you want to use a different LLM provider (examples are in `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (none) | Your Groq key |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI compatible API, e.g. OpenAI or a local Ollama server |
| `LLM_API_KEY` | (none) | Key for that API. Used instead of `GROQ_API_KEY` when set |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name on that API |
