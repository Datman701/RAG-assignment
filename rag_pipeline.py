"""
rag_pipeline.py
---------------
Orchestrates the full RAG pipeline:
  1. Document loading  (document_loader)
  2. Chunking + embedding (vectorstore)
  3. Retrieval + LLM answer generation (this module)

Supported file types: pdf, txt, csv
"""

import logging
import os
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from document_loader import load_document
from vectorstore import OpenRouterEmbeddings, create_chunks, create_vectorstore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt used for answer generation
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the provided "
    "document context. If the answer is not in the context, say: 'I could not find an "
    "answer to that in the uploaded document.' Do not use any outside knowledge."
)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline: ingest → embed → retrieve → generate."""

    def __init__(self) -> None:
        """Initialise local embeddings and the Gemini LLM from environment variables."""
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Configure it in Render environment variables."
            )

        self.gemini_api_key = gemini_api_key
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        embedding_model = os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )

        self.embeddings = OpenRouterEmbeddings(model_name=embedding_model)
        self.llms = self._build_llm_chain(gemini_model)

        logger.info(
            "RAGPipeline initialised with Gemini LLM '%s' and embedding model '%s'",
            self.llms[0][0],
            embedding_model,
        )

    def _build_llm_chain(self, primary_model: str) -> List[Tuple[str, ChatGoogleGenerativeAI]]:
        """Create the primary Gemini model plus a Gemma fallback."""
        model_names = [primary_model]
        fallback_model = "gemma-4-31b-it"
        if fallback_model not in model_names:
            model_names.append(fallback_model)

        llms: List[Tuple[str, ChatGoogleGenerativeAI]] = []
        for model_name in model_names:
            llms.append((
                model_name,
                ChatGoogleGenerativeAI(  # type: ignore
                    model=model_name,
                    temperature=0,
                    google_api_key=self.gemini_api_key
                )
            ))

        return llms

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "429" in message
            or "resource_exhausted" in message
            or "rate limit" in message
            or "quota" in message
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self, file_path: str, file_type: str
    ) -> Tuple[List, str]:
        """
        Load a document from *file_path*.

        Delegates to :func:`document_loader.load_document`.

        Args:
            file_path: Path to the file on disk.
            file_type: ``"pdf"``, ``"txt"``, or ``"csv"`` (case-insensitive).

        Returns:
            ``(documents, file_name)``
        """
        return load_document(file_path, file_type)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_document(
        self,
        file_path: str,
        file_type: str,
        collection_name: str = "documents",
    ) -> Tuple[Chroma, Dict[str, Any]]:
        """
        Run the full ingest → chunk → embed pipeline.

        Args:
            file_path:       Path to the document file.
            file_type:       ``"pdf"``, ``"txt"``, or ``"csv"``.
            collection_name: Unique Chroma collection name for this session.

        Returns:
            ``(vectorstore, metadata)`` where *metadata* contains:
            ``file_name``, ``file_type``, ``chunk_count``, ``total_documents``.
        """
        documents, file_name = load_document(file_path, file_type)
        chunks = create_chunks(documents)
        vectorstore = create_vectorstore(chunks, self.embeddings, collection_name)

        metadata: Dict[str, Any] = {
            "file_name": file_name,
            "file_type": file_type,
            "chunk_count": len(chunks),
            "total_documents": len(documents),
        }
        return vectorstore, metadata

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def answer_question(
        self,
        vectorstore: Chroma,
        question: str,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks and generate an answer with the LLM.

        Args:
            vectorstore: Populated Chroma vectorstore for this session.
            question:    User's natural-language question.

        Returns:
            ``{"answer": str, "sources": list[dict]}``
        """
        try:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )

            context_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in context_docs)

            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Context from document:\n\n{context_text}\n\nQuestion: {question}"
                ),
            ]

            last_error: Exception | None = None
            response = None
            for index, (model_name, llm) in enumerate(self.llms):
                try:
                    response = llm.invoke(messages)
                    if model_name != self.llms[0][0]:
                        logger.info("Answer generated using fallback Gemini model '%s'", model_name)
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("Gemini model '%s' failed: %s", model_name, exc)
                    if index == 0 and self._is_rate_limit_error(exc):
                        continue
                    if index == 0:
                        raise

            if response is None:
                raise last_error or RuntimeError("No Gemini model available for answer generation")

            # Safely extract text from the LLM response
            if hasattr(response, "content"):
                if isinstance(response.content, list):
                    text_parts = [
                        part.get("text", "")
                        for part in response.content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    answer = "".join(text_parts) if text_parts else str(response.content)
                else:
                    answer = str(response.content)
            else:
                answer = str(response)

            sources = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in context_docs
            ]

            logger.info("Answer generated using %d source chunk(s)", len(sources))
            return {"answer": answer, "sources": sources}

        except Exception as exc:
            logger.error("Error generating answer: %s", exc)
            raise ValueError(f"Error generating answer: {exc}") from exc
