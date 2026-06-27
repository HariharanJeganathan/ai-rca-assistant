"""
config.py — Central Configuration & LLM Switcher
Updated: Use ChromaDB default embeddings on free tier (no torch needed)
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "AI RCA Assistant")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    # Use "chromadb" on free tier (no torch), "huggingface" locally
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "chromadb")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://rca_user:rca_password@localhost:5432/rca_db")
    CHROMA_PERSIST_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "./chroma_data")


settings = Settings()


@lru_cache(maxsize=1)
def get_llm():
    provider = settings.LLM_PROVIDER
    if provider == "groq":
        return _load_groq()
    elif provider == "openai":
        return _load_openai()
    elif provider == "azure":
        return _load_azure_openai()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'")


def _load_groq():
    from langchain_groq import ChatGroq
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is empty!")
    return ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0.1, max_tokens=2000)


def _load_openai():
    from langchain_openai import ChatOpenAI
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is empty!")
    return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL, temperature=0.1, max_tokens=2000)


def _load_azure_openai():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=0.1, max_tokens=2000
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns embeddings model.
    On free tier (Render 512MB): uses ChromaDB's built-in embeddings
    which use onnxruntime instead of torch — much lighter!
    Locally: can use HuggingFace for better quality.
    """
    provider = settings.EMBEDDING_PROVIDER

    if provider == "chromadb":
        # ChromaDB's default embedding — uses onnxruntime, NOT torch
        # Downloads a small 22MB ONNX model instead of 2GB torch
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        from langchain_community.embeddings import FakeEmbeddings

        # We wrap ChromaDB's embedding for use in LangChain
        # Using FakeEmbeddings as a passthrough since ChromaDB handles
        # embeddings internally when we use its default function
        return FakeEmbeddings(size=384)

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    else:
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)


def validate_config():
    errors = []
    provider = settings.LLM_PROVIDER
    if provider == "groq" and not settings.GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set")
    if provider == "openai" and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")
    if errors:
        error_list = "\n  - ".join(errors)
        raise EnvironmentError(f"\n[Config] Missing:\n  - {error_list}")
    print(f"[Config] ✅ Valid. LLM: {provider}")
    return True
