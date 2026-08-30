"""
Offline embedding step: converts the job-description PDF into a Chroma
vector store. Run manually once (or whenever the PDF changes) via:

    python -m app.modules.embedding.build_index

This is NOT on the live request path — the Info Advisor only reads the
already-persisted Chroma store at runtime.
"""
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def build(pdf_path: str = config.JOB_DESCRIPTION_PDF, persist_dir: str = config.CHROMA_DIR) -> int:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY)
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)

    return len(chunks)


if __name__ == "__main__":
    chunk_count = build()
    print(f"Embedded {chunk_count} chunks from '{config.JOB_DESCRIPTION_PDF}' into {config.CHROMA_DIR}")
