"""
config.py — Central Configuration & LLM Switcher
Updated: Use ChromaDB default embeddings on free tier (no torch needed)
"""

import logging
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "AI RCA Assistant")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    # Safe fallback for environments where the configured Groq model is
    # unavailable because of model retirement or project permissions.
    GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
    GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "2"))
    GROQ_TIMEOUT_SECONDS: int = int(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # Embeddings
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "chromadb")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Persistence
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


def _resolve_groq_model() -> str:
    """Return a Groq model that this API key can actually access.

    The deployed app previously failed when Render retained an older
    GROQ_MODEL value. We validate the configured model once at LLM startup.
    If it is unavailable to the current Groq project/key, we fall back to
    the known working lightweight production model instead of allowing all
    seven RCA steps to fail.
    """
    requested = settings.GROQ_MODEL.strip()
    fallback = settings.GROQ_FALLBACK_MODEL.strip() or "llama-3.1-8b-instant"

    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        client.models.retrieve(requested)
        logger.info(f"[Config] Groq model available: {requested}")
        return requested
    except Exception as requested_error:
        if requested == fallback:
            raise RuntimeError(
                f"Configured Groq model '{requested}' is unavailable: {requested_error}"
            ) from requested_error

        logger.warning(
            f"[Config] Groq model '{requested}' is unavailable. "
            f"Trying fallback '{fallback}'. Reason: {requested_error}"
        )

        try:
            client.models.retrieve(fallback)
            logger.info(f"[Config] Using Groq fallback model: {fallback}")
            return fallback
        except Exception as fallback_error:
            raise RuntimeError(
                f"Neither Groq model '{requested}' nor fallback '{fallback}' "
                f"is available. Requested error: {requested_error}; "
                f"fallback error: {fallback_error}"
            ) from fallback_error


def _load_groq():
    from langchain_groq import ChatGroq
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is empty!")

    model = _resolve_groq_model()

    # Keep provider retries bounded. Temporary 429s can recover, but a
    # daily quota exhaustion cannot be fixed by repeated requests. The
    # RCA nodes already have controlled fallback/error handling, so a
    # bounded retry policy prevents long retry loops without changing the
    # LangGraph workflow or any RCA step.
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=model,
        temperature=0.1,
        max_tokens=800,
        max_retries=settings.GROQ_MAX_RETRIES,
        timeout=settings.GROQ_TIMEOUT_SECONDS,
    )


def _load_openai():
    from langchain_openai import ChatOpenAI
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is empty!")
    return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL, temperature=0.1, max_tokens=2000)


def _load_azure_openai():
    from langchain_openai import AzureChatOpenAI
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY is empty!")
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT is empty!")
    if not settings.AZURE_OPENAI_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT is empty!")

    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=2000
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns embeddings model.
    On free tier (Render 512MB): ChromaDB itself uses its built-in
    DefaultEmbeddingFunction (ONNX) inside the retriever, so this LangChain
    embedding object is not used by the historical search path.
    """
    provider = settings.EMBEDDING_PROVIDER

    if provider == "chromadb":
        from langchain_community.embeddings import FakeEmbeddings
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
    if provider == "azure":
        if not settings.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is not set")
        if not settings.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is not set")
        if not settings.AZURE_OPENAI_DEPLOYMENT:
            errors.append("AZURE_OPENAI_DEPLOYMENT is not set")

    if errors:
        error_list = "\n  - ".join(errors)
        raise EnvironmentError(f"\n[Config] Missing:\n  - {error_list}")

    print(f"[Config] ✅ Valid. LLM: {provider}")
    return True
