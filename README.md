# AmbedkarGPT-Intern-Task

A small Retrieval-Augmented-Generation (RAG) prototype for the Kalpit AI Intern assignment.

It loads a short speech excerpt (speech.txt), splits into chunks, embeds them locally using
`sentence-transformers/all-MiniLM-L6-v2`, indexes the embeddings into a local ChromaDB, then answers
questions by retrieving relevant chunks and prompting a local Ollama LLM (Mistral 7B).

---

## Requirements
- Python 3.8+
- Reasonable CPU and RAM (embedding step runs locally). Using a GPU will speed embeddings but is not required.
- Ollama installed and `mistral` model pulled locally.

---

## Setup (Linux/macOS)

1. Clone or copy these files into a folder `AmbedkarGPT-Intern-Task/`.

2. Create & activate a virtualenv:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Install Ollama and pull Mistral (Ollama runs the LLM locally):

```bash
# Install script (official)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull mistral model
ollama pull mistral
```

On Windows, follow the official Ollama Windows install instructions from https://ollama.ai.

5. Run the app (first run will build the vectorstore):

```bash
python main.py
```

To force reindexing (rebuild embeddings & chroma DB):

```bash
python main.py --reindex
```

---

## Usage
Start the app and type questions at the prompt. Example questions:
- "What is the real remedy against caste?"
- "What is the real enemy?"

If the answer is not contained in the speech text, the assistant will respond with: "I don't know based on the provided text." (or similar), because the prompt forces it to rely only on the provided context.

---

## Notes & troubleshooting
- **Ollama not found / mistral not pulled**: make sure `ollama` is installed and `ollama list` shows `mistral`.
- **Slow embeddings**: embedding the model uses CPU — give it a little time. If run on a very low-RAM machine, consider increasing swap or running on a machine with more memory.
- **Chroma storage**: the chroma DB is persisted in `./data/chroma/` by default. Delete this folder if you want to force a fresh build.

---

## Deliverables
- `main.py` — working CLI app
- `requirements.txt` — pip dependencies
- `README.md` — setup and run instructions
- `speech.txt` — provided speech excerpt

## Ask questions:
Question: What does Ambedkar describe as the real enemy?
Answer: The real enemy is the belief in the sanctity of the shastras.