"""
vectorstore.py
--------------
Handles text chunking and ChromaDB vector store creation.

Responsibilities:
    - Split LangChain Documents into fixed-size overlapping chunks.
        - Embed chunks via OpenRouter API.
    - Persist chunks in an in-memory Chroma collection.
"""

import logging
import os
from typing import Any, List, cast

import openrouter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 20       # documents per embedding API call
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embeddings adapter
# ---------------------------------------------------------------------------

class OpenRouterEmbeddings(Embeddings):
    """LangChain embeddings wrapper around OpenRouter API."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, api_key: str = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Configure it in environment variables."
            )
        self.client = openrouter.OpenRouter(api_key=self.api_key)
        logger.info("Initialized OpenRouter embeddings with model '%s'", model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = []
        for text in texts:
            try:
                response: Any = self.client.embeddings.generate(
                    model=self.model_name,
                    input=text,
                )

                # Support both mapping-style and attribute-style responses
                if isinstance(response, dict):
                    embedding_raw = response.get("data", [None])[0]
                    embedding = embedding_raw.get("embedding") if embedding_raw else None
                else:
                    first = getattr(response, "data", None)
                    if isinstance(first, (list, tuple)) and len(first) > 0:
                        embedding = getattr(first[0], "embedding", None)
                    else:
                        embedding = None

                if embedding is None:
                    raise ValueError("No embedding returned from OpenRouter")

                embeddings.append(cast(List[float], embedding))
            except Exception as exc:
                logger.error("Failed to embed text: %s", exc)
                raise ValueError(f"Error embedding text: {exc}") from exc

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            response: Any = self.client.embeddings.generate(
                model=self.model_name,
                input=text,
            )

            if isinstance(response, dict):
                embedding_raw = response.get("data", [None])[0]
                embedding = embedding_raw.get("embedding") if embedding_raw else None
            else:
                first = getattr(response, "data", None)
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    embedding = getattr(first[0], "embedding", None)
                else:
                    embedding = None

            if embedding is None:
                raise ValueError("No embedding returned from OpenRouter")

            return cast(List[float], embedding)
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            raise ValueError(f"Error embedding query: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_chunks(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into overlapping chunks.

    Args:
        documents: Source documents produced by ``document_loader.load_document``.

    Returns:
        A list of chunked Documents, each annotated with a ``chunk_index``
        field in its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))
    return chunks


def create_vectorstore(
    chunks: List[Document],
    embeddings: Embeddings,
    collection_name: str = "documents",
) -> Chroma:
    """
    Embed *chunks* and store them in an in-memory Chroma collection.

    Embedding is done in batches of :data:`EMBED_BATCH_SIZE` to keep memory
    usage predictable when indexing larger documents.

    Args:
        chunks:          Chunked Documents (output of :func:`create_chunks`).
        embeddings:      A configured LangChain :class:`~langchain_core.embeddings.Embeddings` instance.
        collection_name: Name of the Chroma collection; use a unique value per
                         session to prevent data bleeding between uploads.

    Returns:
        A populated :class:`~langchain_community.vectorstores.Chroma` instance.

    Raises:
        ValueError: If embedding fails after all retries.
    """
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
        )

        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            vectorstore.add_documents(batch)

        logger.info("Vectorstore '%s' created with %d chunk(s)", collection_name, len(chunks))
        return vectorstore

    except Exception as exc:
        logger.error("Failed to create vectorstore: %s", exc)
        raise ValueError(f"Error creating vector embeddings: {exc}") from exc
