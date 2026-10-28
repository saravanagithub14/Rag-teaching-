# RAG Notes

Study notes on Retrieval Augmented Generation (RAG). Read these alongside `rag_chatbot.py`: every idea here points to the part of the code where it happens. For hands-on exercises, see `README.md`.

---

## 1. What is RAG?

**RAG = Retrieval Augmented Generation.**

> Before asking the LLM a question, **find** the relevant text in your own documents and **paste it into the prompt**. The LLM then answers from that text instead of from memory.

The three words describe the three moves:

| Word | Meaning | In our code |
|---|---|---|
| **Retrieval** | Search your documents for the pieces related to the question | `retrieve()` |
| **Augmented** | Add those pieces to the prompt | `build_messages()` |
| **Generation** | The LLM writes the answer using them | `ask_llm()` |

**Analogy:** a plain LLM takes a closed book exam: it can only use what it memorised during training. RAG turns it into an open book exam: before each answer, it is handed the right pages. The model itself is not changed or retrained at all.

---

## 2. Why do we need RAG?

LLMs are powerful but have four built-in problems:

1. **Knowledge cutoff.** A model only knows what was in its training data, up to a certain date. It doesn't know today's prices or last week's policy change.
2. **No private knowledge.** It has never seen your company's documents, your syllabus or your cafe's menu.
3. **Hallucination.** When it doesn't know, it often guesses with full confidence. A wrong answer that sounds right is worse than "I don't know".
4. **No sources.** A plain answer can't tell you where it came from, so you can't check it.

RAG addresses all four: the documents can be updated any time, they can be private, the model is told to answer only from them, and we know exactly which chunks were used.

**See it yourself:** ask *"What time does Marina Bean Cafe close?"* with `python rag_chatbot.py --no-rag`. The model has never heard of the cafe, so it either refuses or guesses. With RAG on, it answers correctly from `about_and_policies.txt`.

---

## 3. The two phases

A RAG system works in two separate phases.

```
 INDEXING PHASE  (once, at startup)

   documents --> [1 load] --> [2 chunk] --> [3 embed] --> vector store
                                                              |
 QUERY PHASE  (for every question)                            |
                                                              v
   question --> [4 embed question] --> [5 retrieve top-k] <---+
                                              |
                                              v
                                   [6 build prompt] --> [7 LLM] --> answer
```

| Phase | Steps | When it runs | Cost |
|---|---|---|---|
| **Indexing** | 1 load, 2 chunk, 3 embed | Once, before any questions | Slow, but done once |
| **Query** | 4 embed question, 5 retrieve, 6 build prompt, 7 generate | Every question | Must be fast |

A common mistake is to think everything reruns for each question. It doesn't: the documents are embedded once, and each question only needs **one** new embedding (its own).

---

## 4. The pipeline, step by step

### Step 1: Load

Read the raw text of every document, and keep the **file name** with it so we always know where a piece of text came from.

- **In our code:** `load_documents()` reads every `.txt` and `.md` file in `data/`.
- **In real projects:** PDFs, Word files, web pages and databases each need a text extraction step first (e.g. the `pypdf` library for PDFs). Messy extraction (broken tables, headers on every page) hurts everything after it.

### Step 2: Chunk

Cut each document into small pieces called **chunks**.

**Why not embed a whole document?**
- An embedding of a long document is a blurry average of all its topics. A small chunk has one clear meaning, so search is sharper.
- The LLM prompt has a size limit, and every extra token costs time and money.

**Two settings control chunking:**

| Setting | Our value | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Characters repeated between neighbouring chunks |

The window moves forward by `CHUNK_SIZE - CHUNK_OVERLAP` = 400 characters each time:

```
text:     |----------------------------------------------------------|
chunk 1:  [0 ........... 500]
chunk 2:            [400 ........... 900]
chunk 3:                      [800 ........... 1300]
                    |<-100->| overlap: this text appears in both chunks
```

**Why overlap?** Without it, a sentence cut in half at a boundary is split between two chunks, and neither half makes sense alone. With overlap, it appears whole in at least one chunk.

**The size trade-off:**

| Chunks too small | Chunks too big |
|---|---|
| Lose surrounding context ("It costs 40 rupees": what does?) | Meaning gets diluted, scores become less sharp |
| Need a higher top-k to collect the full answer | Fewer chunks fit in the prompt; more irrelevant text |

**Chunking strategies**, from simplest to smartest:

| Strategy | How it splits | Notes |
|---|---|---|
| Fixed size | Every N characters | **What we use.** Simple, but cuts mid-word and mid-sentence |
| Sentence / paragraph | On `.` or blank lines | Keeps ideas whole; chunk sizes vary |
| Recursive | Tries paragraphs, then sentences, then words, until pieces are small enough | A popular default in RAG libraries |
| By document structure | On headings, sections, slides | Great for manuals and Markdown |
| Semantic | Starts a new chunk where the topic changes | Uses embeddings to find topic shifts; slower |

**Our data:** with the default settings, `about_and_policies.txt` (about 1,340 characters) becomes 4 chunks and `menu_and_prices.txt` (about 790 characters) becomes 2, for 6 in total.

### Step 3: Embed

An **embedding** turns a piece of text into a list of numbers (a **vector**) that represents its **meaning**.

- Our model, `all-MiniLM-L6-v2`, produces **384 numbers** per text.
- Texts with similar meaning get similar vectors, **even when they share no words**. "Refund" and "get my money back" end up close together; "refund" and "coffee" end up far apart.
- This is what makes **semantic search** possible: searching by meaning, not by matching keywords.

Think of each embedding as a point on a map with 384 directions instead of 2. Similar meanings are neighbours on that map.

```
                     "can I get my money back?"
                       *  "refund policy"
                      *
                                                  * "Cappuccino 120 rupees"
                                                 *  "price of coffee"
          * "dogs on the terrace"
         *  "can I bring my pet?"
```

**Normalising.** `normalize_embeddings=True` scales every vector to length 1. Only its direction (its meaning) is kept. This makes the similarity calculation in step 5 simpler.

**The result** is a numpy array of shape `(number_of_chunks, 384)`: one row per chunk. For our data that is `(6, 384)`. **That array is our vector store.**

- **In our code:** `build_index()`.

### Step 4: Embed the question

The question is embedded with the **same model** as the chunks.

Different models put vectors on completely different "maps". Comparing a vector from model A with one from model B is like comparing a distance in centimetres with one in inches without converting: the numbers mean nothing together.

- **In our code:** the first line of `retrieve()`.

### Step 5: Retrieve

Score every chunk against the question, then keep the best **k** (`TOP_K = 3`).

**Cosine similarity** measures how much two vectors point in the same direction:

```
                    A · B
cosine(A, B) = -------------
                |A| × |B|
```

| Score | Meaning |
|---|---|
| 1.0 | Same direction: same meaning |
| around 0 | Unrelated |
| -1.0 | Opposite direction |

Because we normalised every vector to length 1, `|A| × |B| = 1`, so **cosine similarity is just the dot product** `A · B`. That is why the code only needs `np.dot(chunk_vector, question_vector)`.

**Worked example** with tiny 3-number vectors (real ones have 384):

```
question   Q = [1/3,  2/3,  2/3]      (length 1)
chunk A    A = [2/3,  1/3,  2/3]      (length 1)
chunk B    B = [2/3, -2/3,  1/3]      (length 1)

Q · A = (1/3)(2/3) + (2/3)(1/3) + (2/3)(2/3) = 2/9 + 2/9 + 4/9 = 8/9 ≈ 0.89   -> very similar
Q · B = (1/3)(2/3) + (2/3)(-2/3) + (2/3)(1/3) = 2/9 - 4/9 + 2/9 = 0          -> unrelated
```

So chunk A would be retrieved first.

**Choosing k:**

| k too small | k too large |
|---|---|
| Misses answers that are spread across several chunks | Adds irrelevant text, which can confuse the model |
| One bad ranking means a wrong answer | Longer, slower, more expensive prompts |

**Reading scores in our project.** With such small documents, a good match usually scores around 0.3 to 0.5, and the best chunk is not always ranked first. Scores are **relative**: compare them to each other, not to a fixed "good" number. A question with no answer in the documents (like "Do you sell pizza?") gets scores that are all low and close together.

- **In our code:** the loop in `retrieve()`, then `results.sort(key=get_score, reverse=True)` and `results[:k]`.

### Step 6: Build the prompt (augment)

Paste the retrieved chunks into the prompt, above the question. Each chunk is labelled with its source file:

```
Context:
[Source: about_and_policies.txt]
...Pets Well behaved dogs are welcome on the terrace, never inside...

[Source: ...]
...

Question: Can I bring my dog inside?
```

The **system prompt** sets the rules. Two instructions do most of the work:

1. Answer using **ONLY** the context provided.
2. If the answer is not in the context, **say you don't have that information**.

Without them, the model mixes in its own knowledge and guesses, and the point of RAG is lost. This is called **grounding**: keeping the answer tied to the given sources.

- **In our code:** `SYSTEM_PROMPT` and `build_messages()`.

### Step 7: Generate

Send the messages to the LLM and return its answer.

- We use `temperature=0` so the model always picks its most likely words. For factual questions we want steady, repeatable answers, not creativity.
- Groq offers an **OpenAI compatible API**, so the standard `openai` Python library works by changing only the base URL. The same code can talk to OpenAI, Groq or a local Ollama server.
- **In our code:** `ask_llm()`.

---

## 5. RAG compared with other approaches

| Approach | How it adds knowledge | Good for | Weak at |
|---|---|---|---|
| **Plain LLM** | Nothing extra | General knowledge, writing | Private or recent facts |
| **RAG** | Looks up documents for every question | Facts that change; private documents; answers you need to check | Questions that need the whole document at once |
| **Fine-tuning** | Retrains the model's weights on your data | Changing style, tone or format; specialised tasks | Adding facts that change; showing sources; it is costly to redo |
| **Long context** | Pastes whole documents into every prompt | A few documents, one-off questions | Large collections; every question pays for all the text |

**Rule of thumb:** use RAG to give the model **knowledge**, and fine-tuning to change its **behaviour**. They can be combined.

---

## 6. Where RAG goes wrong

When an answer is wrong, always ask first: **did retrieval fail, or did generation fail?** Look at the retrieved chunks to decide.

| Failure | What you see | Typical fix |
|---|---|---|
| **Retrieval miss** | The chunk with the answer isn't in the top-k list | Change chunk size, raise k, use hybrid search or reranking |
| **Answer split across chunks** | Each chunk has part of the answer, but not all are retrieved | Raise k, use bigger chunks or more overlap |
| **Bad chunk boundary** | The key sentence is cut in half | More overlap, sentence or paragraph chunking |
| **Generation error** | The right chunk is there, but the answer is still wrong | Clearer prompt, a stronger model |
| **Hallucination** | The answer includes facts not in any chunk | Stricter system prompt; a minimum score threshold |
| **No-answer questions** | The bot answers even when the documents don't cover it | Refuse when all scores are below a threshold |
| **Stale data** | The answer is from an old version of a document | Rebuild the index when documents change |
| **Lost follow-ups** | "And with oat milk?" fails | Add chat memory and rewrite follow-ups into full questions |

---

## 7. Measuring a RAG system

Don't judge a RAG system by trying a few questions by eye. Build a **test set**: a list of questions with known correct answers and the chunk or file that contains each answer. (The answer key in `README.md` is a small test set.)

**Retrieval metrics** (did we find the right chunks?):

| Metric | Question it answers |
|---|---|
| **Hit rate / recall@k** | How often is the correct chunk somewhere in the top k? |
| **MRR** (mean reciprocal rank) | How high up is the correct chunk? First place scores 1, second 1/2, third 1/3 |
| **Precision@k** | Of the k chunks returned, how many were actually relevant? |

**Generation metrics** (did we use them well?):

| Metric | Question it answers |
|---|---|
| **Faithfulness / groundedness** | Is every claim in the answer supported by the retrieved chunks? |
| **Answer relevance** | Does the answer actually address the question? |
| **Correctness** | Does it match the known correct answer? |

Measure retrieval first. If the right chunk is not retrieved, no prompt can fix the answer.

---

## 8. Making RAG better

Techniques used in real systems, roughly in order of how often they're used:

- **Better chunking.** Split on sentences, paragraphs or headings instead of a fixed character count.
- **Metadata filtering.** Store extra fields with each chunk (file, date, department, language) and filter before searching, e.g. "only search the 2026 policy files".
- **Hybrid search.** Combine semantic search (embeddings) with keyword search (such as BM25). Keywords catch exact terms that embeddings can miss: product codes, names, error numbers.
- **Reranking.** Retrieve many candidates (say 20) quickly, then use a slower, more accurate model (a cross-encoder) to pick the best 3.
- **Query rewriting.** Rewrite a vague or follow-up question into a clear standalone one before searching ("And with oat milk?" becomes "What is the price of a cappuccino with oat milk?").
- **Multi-query.** Generate several rephrasings of the question, search with each, and merge the results.
- **Small-to-big retrieval.** Search over small chunks for precision, but give the LLM the larger section around the best match for context.
- **Citations.** Ask the model to name the source of each claim so users can check it.
- **Score threshold.** If nothing scores above a minimum, reply "I don't know" without calling the LLM.
- **Agentic RAG.** Let the LLM decide when to search, what to search for, and whether to search again after seeing the results.

---

## 9. Vector stores

A **vector store** keeps the embeddings and finds the closest ones to a query vector.

- **Ours is a numpy array.** It compares the question with every chunk: a **brute force** or **exact** search. That is perfectly fast for hundreds or thousands of chunks.
- **With millions of chunks**, comparing with every one is too slow. Vector databases use **approximate nearest neighbour (ANN)** indexes, such as HNSW, that find *almost* the best matches very quickly by skipping most comparisons.
- They also save the index to disk, handle updates and deletes, and support metadata filtering.

| Tool | Type |
|---|---|
| numpy array | What we use. In memory, exact search |
| FAISS | Fast similarity search library (in memory) |
| Chroma | Simple, local vector database |
| pgvector | Vector search inside PostgreSQL |
| Pinecone, Weaviate, Qdrant, Milvus | Dedicated vector databases, local or hosted |

A vector database does not change *what* RAG is. It is the same idea as our numpy array, built to handle much more data.

---

## 10. Our project at a glance

| Concept | Function in `rag_chatbot.py` | Setting |
|---|---|---|
| Load documents | `load_documents()` | `DATA_DIR` |
| Chunking | `chunk_text()`, `build_chunks()` | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Embedding and vector store | `build_index()` | `EMBED_MODEL_NAME` |
| Question embedding and retrieval | `retrieve()` | `TOP_K` |
| Prompt and grounding | `build_messages()` | `SYSTEM_PROMPT` |
| Generation | `ask_llm()` | `LLM_MODEL`, `LLM_BASE_URL` (in `.env`) |

**Key numbers:**

| Item | Value |
|---|---|
| Documents | 2 files in `data/` |
| Chunks (default settings) | 6 |
| Embedding size | 384 numbers per chunk |
| Vector store shape | `(6, 384)` |
| Chunks sent to the LLM | 3 (`TOP_K`) |
| Temperature | 0 |

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large Language Model. A model that generates text, such as GPT or Llama |
| **RAG** | Retrieval Augmented Generation. Look things up, then answer |
| **Hallucination** | The model confidently stating something false |
| **Knowledge cutoff** | The date after which a model has no training data |
| **Corpus** | The collection of documents you search over |
| **Chunk** | A small piece of a document; the unit we search over |
| **Overlap** | Text repeated between neighbouring chunks |
| **Token** | A piece of a word that LLMs read and write; prompts are measured and charged in tokens |
| **Embedding** | A list of numbers representing the meaning of a text |
| **Vector** | A list of numbers; an embedding is a vector |
| **Dimension** | How many numbers are in a vector (384 here) |
| **Normalise** | Scale a vector to length 1 so only its direction matters |
| **Cosine similarity** | How closely two vectors point the same way, from -1 to 1 |
| **Dot product** | Multiply matching numbers and add them up; equals cosine similarity for length-1 vectors |
| **Semantic search** | Searching by meaning instead of exact words |
| **Keyword search / BM25** | Searching by matching words; BM25 is the classic scoring method |
| **Hybrid search** | Semantic and keyword search combined |
| **Vector store / database** | Where embeddings are kept and searched |
| **ANN** | Approximate nearest neighbour; fast, almost-exact vector search |
| **Top-k** | The k best matching chunks |
| **Reranking** | Re-ordering retrieved chunks with a more accurate model |
| **Context** | The retrieved text placed in the prompt |
| **Context window** | The maximum amount of text a model can read at once |
| **System prompt** | Standing instructions that tell the model how to behave |
| **Grounding** | Keeping the answer tied to the provided sources |
| **Temperature** | How random the output is; 0 is the most predictable |
| **Fine-tuning** | Further training a model's weights on new data |

---

## 12. Check yourself

Try answering each question before opening it.

<details>
<summary>1. In one sentence, what does RAG do?</summary>

It retrieves relevant pieces of your own documents and adds them to the prompt, so the LLM answers from them instead of from memory.
</details>

<details>
<summary>2. Which steps run once, and which run for every question?</summary>

Load, chunk and embed the documents run once (indexing). Embedding the question, retrieving, building the prompt and calling the LLM run for every question.
</details>

<details>
<summary>3. Why do we split documents into chunks?</summary>

A small chunk has one clear meaning, so its embedding is sharper and search is more accurate. Chunks also keep the prompt short, which keeps it within the model's limit and costs less.
</details>

<details>
<summary>4. With CHUNK_SIZE = 500 and CHUNK_OVERLAP = 100, where does the third chunk start?</summary>

At character 800. The window moves 500 - 100 = 400 characters each time: 0, 400, 800.
</details>

<details>
<summary>5. Why must the question be embedded with the same model as the chunks?</summary>

Each model creates its own "map" of meaning. Vectors from different models can't be compared, just as centimetres and inches can't be compared without converting.
</details>

<details>
<summary>6. Why is a dot product enough to calculate cosine similarity in our code?</summary>

All vectors are normalised to length 1, so the bottom of the cosine formula (|A| × |B|) equals 1, leaving just the dot product.
</details>

<details>
<summary>7. The right chunk was retrieved, but the answer is still wrong. Retrieval problem or generation problem?</summary>

Generation. Retrieval did its job; look at the prompt or the model.
</details>

<details>
<summary>8. What happens if you remove "ONLY" from the system prompt?</summary>

The model starts mixing in its own knowledge and guesses, so it may answer questions the documents don't cover, and may hallucinate.
</details>

<details>
<summary>9. Would you use RAG or fine-tuning to teach a chatbot this week's prices? Why?</summary>

RAG. Prices change often; with RAG you just update the document. Fine-tuning would need retraining every time and can't show which source an answer came from.
</details>

<details>
<summary>10. When would you replace our numpy array with a vector database?</summary>

When there are too many chunks for brute force search to be fast (many thousands to millions), or when you need to save the index, update it often, or filter by metadata.
</details>
