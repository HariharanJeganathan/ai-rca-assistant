"""
config.py — Central Configuration & LLM Switcher
==================================================
This file reads your .env file and returns the correct
LLM (AI model) based on what you set in LLM_PROVIDER.

HOW IT WORKS:
    .env file says: LLM_PROVIDER=groq
    → config.py returns a Groq LLM object
    → All other files just call get_llm() and don't care which one

    Change .env to: LLM_PROVIDER=openai
    → Now the whole app uses OpenAI. No code changes needed!

This pattern is called "Dependency Injection" — a best practice
that senior engineers use and interviewers love to see.
"""

import os
from functools import lru_cache          # Cache so we don't re-create LLM every request
from dotenv import load_dotenv           # Reads .env file into environment variables

# Load .env file (must be in project root)
load_dotenv()


# ============================================================
# Settings Class — holds all config values
# ============================================================
class Settings:
    """
    All app settings in one place.
    Values come from environment variables (.env file).
    os.getenv("KEY", "default") means: read KEY, use "default" if not found.
    """

    # App info
    APP_NAME: str = os.getenv("APP_NAME", "AI RCA Assistant")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Which LLM provider to use
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

    # Groq settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # Embeddings (for ChromaDB)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://rca_user:rca_password@localhost:5432/rca_db"
    )

    # ChromaDB
    CHROMA_PERSIST_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "./chroma_data")


# Create a single instance (singleton pattern)
settings = Settings()


# ============================================================
# LLM Factory — returns the right AI model
# ============================================================
@lru_cache(maxsize=1)   # Only create the LLM object once, reuse it
def get_llm():
    """
    Returns a LangChain LLM object based on LLM_PROVIDER in .env

    Usage (in any other file):
        from config import get_llm
        llm = get_llm()
        response = llm.invoke("Explain this incident...")

    The calling code doesn't need to know if it's Groq or OpenAI!
    """

    provider = settings.LLM_PROVIDER
    print(f"[Config] Loading LLM provider: {provider}")

    if provider == "groq":
        return _load_groq()

    elif provider == "openai":
        return _load_openai()

    elif provider == "azure":
        return _load_azure_openai()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Choices are: groq, openai, azure"
        )


def _load_groq():
    """Load Groq LLM (free tier, LLaMA 3)"""
    try:
        from langchain_groq import ChatGroq

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is empty! "
                "Get your free key at https://console.groq.com"
            )

        print(f"[Config] Using Groq model: {settings.GROQ_MODEL}")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.1,    # Low = more factual, less creative (good for RCA)
            max_tokens=2000,
        )
    except ImportError:
        raise ImportError(
            "langchain-groq not installed. Run: pip install langchain-groq"
        )


def _load_openai():
    """Load OpenAI LLM"""
    try:
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is empty! "
                "Get your key at https://platform.openai.com/api-keys"
            )

        print(f"[Config] Using OpenAI model: {settings.OPENAI_MODEL}")
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.1,
            max_tokens=2000,
        )
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. Run: pip install langchain-openai"
        )


def _load_azure_openai():
    """Load Azure OpenAI LLM"""
    try:
        from langchain_openai import AzureChatOpenAI

        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is empty!")

        print(f"[Config] Using Azure OpenAI deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=2000,
        )
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. Run: pip install langchain-openai"
        )


# ============================================================
# Embeddings Factory — returns the right embedding model
# ============================================================
@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns a LangChain Embeddings object for ChromaDB.
    Embeddings = converting text into numbers so AI can search it.

    HuggingFace is used by default (free, runs locally).
    """

    provider = settings.EMBEDDING_PROVIDER
    print(f"[Config] Loading embeddings provider: {provider}")

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
                # "all-MiniLM-L6-v2" is small, fast, and free
            )
        except ImportError:
            raise ImportError(
                "langchain-huggingface not installed. "
                "Run: pip install langchain-huggingface sentence-transformers"
            )

    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("pip install langchain-openai")

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: '{provider}'")


# ============================================================
# Validation — run this at startup to catch missing keys early
# ============================================================
def validate_config():
    """
    Call this when the app starts.
    Checks that required environment variables are set.
    Better to fail fast at startup than crash mid-request!
    """
    errors = []

    provider = settings.LLM_PROVIDER

    if provider == "groq" and not settings.GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set (required when LLM_PROVIDER=groq)")

    if provider == "openai" and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set (required when LLM_PROVIDER=openai)")

    if provider == "azure":
        if not settings.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is not set")
        if not settings.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is not set")
        if not settings.AZURE_OPENAI_DEPLOYMENT:
            errors.append("AZURE_OPENAI_DEPLOYMENT is not set")

    if errors:
        error_list = "\n  - ".join(errors)
        raise EnvironmentError(
            f"\n[Config] Missing required environment variables:\n  - {error_list}\n"
            f"Copy .env.example to .env and fill in the values."
        )

    print(f"[Config] ✅ Configuration valid. Using LLM: {provider}")
    return True
