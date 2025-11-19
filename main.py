from typing import Union, List
from pathlib import Path
import argparse
import os
import sys
import textwrap


try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama
except Exception as e:
    print("ERROR: required packages missing. Install inside venv:")
    print("  pip install langchain langchain-community langchain-text-splitters chromadb langchain-chroma sentence-transformers transformers")
    raise

BASE_DIR = Path(__file__).parent.resolve()
SPEECH_PATH = BASE_DIR / "speech.txt"
PERSIST_DIR = BASE_DIR / "data" / "chroma"

PROMPT_TEMPLATE = textwrap.dedent("""
You are an assistant that answers user questions ONLY using the provided context.
Do NOT use any outside knowledge. If the answer is not contained in the context, respond exactly:
"I don't know based on the provided text."

Instructions:
- Answer in 1 or 2 short sentences (maximum).
- Do NOT repeat the context verbatim; synthesize and be concise.
- If the answer is plainly stated in the context, give a short paraphrase (not a copy).

Context:
{context}

Question: {question}

Answer concisely:
""").strip()


def build_vectorstore(speech_path: Union[str, Path], persist_dir: Union[str, Path],
                      chunk_size: int = 400, chunk_overlap: int = 40):
    speech_path = Path(speech_path)
    persist_dir = Path(persist_dir)

    if not speech_path.exists():
        print(f"ERROR: speech file not found at {speech_path}")
        sys.exit(1)

    print("Loading document...")
    loader = TextLoader(str(speech_path), encoding="utf-8")
    docs = loader.load()

    print(
        f"Splitting into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    print("Creating embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Indexing {len(chunks)} document chunks into ChromaDB at {persist_dir} ...")
    chroma = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(persist_dir))
    try:
        chroma.persist()
    except Exception:
        pass

    print("Done indexing.")
    return chroma


def load_or_build(speech_path: Union[str, Path], persist_dir: Union[str, Path], reindex: bool = False):
    persist_dir = Path(persist_dir)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not reindex and persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"Loading existing ChromaDB from {persist_dir} ...")
        try:
            chroma = Chroma(persist_directory=str(persist_dir),
                            embedding_function=embeddings)
            print("Loaded Chroma DB.")
            return chroma
        except Exception as e:
            print("Failed to load existing Chroma DB (will rebuild). Error:", e)

    return build_vectorstore(speech_path, persist_dir)


def create_llm(model_name: str = "mistral", temperature: float = 0.0):
    """
    Create an Ollama LLM object using the langchain_community Ollama wrapper.
    """
    return Ollama(model=model_name, temperature=temperature)


def call_llm(llm, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
    """
    Call llm.generate([prompt]) and extract text from the returned LLMResult.
    Returns the generated text (string) or raises RuntimeError if extraction fails.
    """
    
    try:
        gen_result = llm.generate([prompt], stop=stop, **kwargs)
    except Exception as e:
        raise RuntimeError(f"llm.generate() raised an exception: {e}")

    
    try:
        gens = getattr(gen_result, "generations", None)
        if gens and len(gens) > 0 and len(gens[0]) > 0:
            candidate = gens[0][0]
       
            for attr in ("text", "generation_text", "content"):
                if hasattr(candidate, attr):
                    txt = getattr(candidate, attr)
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
            s = str(candidate).strip()
            if s:
                return s
    except Exception:
        pass

    try:
        raw = getattr(gen_result, "raw", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    except Exception:
        pass

    raise RuntimeError("Could not extract generated text from LLMResult (inspect .generations).")


def retrieve_context(chroma: Chroma, query: str, k: int = 1) -> List[str]:
    """
    Return list of retrieved chunk texts (strings). Uses retriever if available,
    otherwise falls back to direct similarity search.
    """
    try:
        retriever = chroma.as_retriever(
            search_type="similarity", search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
    except Exception:
        try:
            docs = chroma.similarity_search(query, k=k)
        except Exception as e:
            print("Retrieval error:", e)
            docs = []

    texts = []
    for d in docs:
        if hasattr(d, "page_content"):
            texts.append(d.page_content)
        else:
            texts.append(str(d))
    return texts


def produce_answer_from_llm(llm, context_texts: List[str], question: str) -> str:
    """
    Build the prompt from the top-k contexts and call the llm.
    Post-process to produce 1-2 sentences.
    """
    if not context_texts:
        return "I don't know based on the provided text."

    context_joined = "\n\n".join(context_texts)
    max_ctx = 2000
    if len(context_joined) > max_ctx:
        context_joined = context_joined[:max_ctx].rsplit(" ", 1)[0] + "..."

    prompt = PROMPT_TEMPLATE.format(context=context_joined, question=question)

    try:
        raw = call_llm(llm, prompt)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    out = (raw or "").strip()

    if not out:
        return "I don't know based on the provided text."
    if len(out) > 400:
        parts = out.split(". ")
        if len(parts) >= 1:
            out = parts[0].strip()
            if len(parts) >= 2:
                out = (parts[0].strip() + ". " + parts[1].strip())
    
    if len(out) > 400:
        out = out[:400].rsplit(" ", 1)[0] + "..."
    return out


def cli_loop(chroma: Chroma, llm):
    print("\nAmbedkarGPT - Q&A (type 'exit' or Ctrl+C to quit)\n")
    show_retrieved = os.environ.get("SHOW_RETRIEVED", "") == "1"

    try:
        while True:
            q = input("Question: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break

            
            contexts = retrieve_context(chroma, q, k=1)
            if show_retrieved:
                print("\n--- Retrieved (debug) ---")
                for i, c in enumerate(contexts):
                    print(f"[{i}] {c[:700].replace(chr(10),' ')}...")
                print("--- end retrieved ---\n")

            try:
                ans = produce_answer_from_llm(llm, contexts, q)
            except Exception as e:
                print("LLM error:", e)
                print("Falling back to returning the retrieved context.")
                if contexts:
                    ans = contexts[0][:600].strip(
                    ) + ("..." if len(contexts[0]) > 600 else "")
                else:
                    ans = "I don't know based on the provided text."

            print("\nAnswer:\n", ans, "\n")
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true",
                        help="Rebuild embeddings & Chroma index")
    parser.add_argument("--speech", type=str,
                        default=str(SPEECH_PATH), help="Path to speech.txt")
    parser.add_argument("--persist-dir", type=str,
                        default=str(PERSIST_DIR), help="Chroma persist dir")
    parser.add_argument("--mistral", type=str,
                        default="mistral", help="Ollama model name")
    args = parser.parse_args()

    
    if os.environ.get("OLLAMA_HOST", "") == "":
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

    chroma = load_or_build(args.speech, args.persist_dir, reindex=args.reindex)

    llm = create_llm(model_name=args.mistral, temperature=0.0)

    cli_loop(chroma, llm)


if __name__ == "__main__":
    main()
