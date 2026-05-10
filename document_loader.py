"""
document_loader.py
------------------
Handles loading and parsing of supported document types:
  - PDF  (.pdf)
  - Plain text (.txt)
  - CSV  (.csv)

Returns LangChain Document objects ready for chunking.
"""

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import PyPDF2
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(file_path: str, file_type: str, override_name: str = None) -> Tuple[List[Document], str]:
    """
    Load and parse a document from disk.

    Args:
        file_path: Absolute or relative path to the file.
        file_type:  One of ``"pdf"``, ``"txt"``, or ``"csv"`` (case-insensitive).
        override_name: Optional original filename to use in metadata.

    Returns:
        A tuple of ``(documents, file_name)`` where *documents* is a non-empty
        list of :class:`langchain_core.documents.Document` objects.

    Raises:
        ValueError: If the file type is unsupported, the file cannot be read,
                    or no text content can be extracted.
    """
    file_name = override_name if override_name else Path(file_path).name
    ft = file_type.lower()

    try:
        if ft == "pdf":
            documents = _load_pdf(file_path, file_name)
        elif ft == "txt":
            documents = _load_txt(file_path, file_name)
        elif ft == "csv":
            documents = _load_csv(file_path, file_name)
        else:
            raise ValueError(f"Unsupported file type: '{file_type}'. Allowed: pdf, txt, csv.")

        if not documents:
            raise ValueError("No text content could be extracted from the document.")

        logger.info("Loaded %d document(s) from '%s'", len(documents), file_name)
        return documents, file_name

    except PyPDF2.errors.PdfReadError as exc:
        logger.error("Failed to parse PDF '%s': %s", file_name, exc)
        raise ValueError(
            "Could not extract text from PDF. Please upload a text-based (non-scanned) PDF."
        ) from exc
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Unexpected error loading '%s': %s", file_name, exc)
        raise ValueError(f"Error processing document: {exc}") from exc


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_pdf(file_path: str, file_name: str) -> List[Document]:
    """Extract text page-by-page from a PDF file."""
    documents: List[Document] = []
    with open(file_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_name, "page": page_num + 1, "type": "pdf"},
                    )
                )
    return documents


def _load_txt(file_path: str, file_name: str) -> List[Document]:
    """Read a UTF-8 plain-text file as a single Document."""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Could not decode text file. Please ensure it is UTF-8 encoded."
        ) from exc

    documents: List[Document] = []
    if text.strip():
        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_name, "type": "txt"},
            )
        )
    return documents


def _detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file by trying common encodings.

    Returns the name of the detected encoding, or 'utf-8' as fallback.
    """
    encodings = ['utf-8', 'utf-16', 'latin-1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # Try to read a sample
            logger.info("Detected encoding: %s", encoding)
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Final fallback
    logger.warning("Could not detect encoding, using utf-8 (may fail)")
    return 'utf-8'


def _load_csv(file_path: str, file_name: str) -> List[Document]:
    """
    Load a CSV file.

    Each row becomes its own Document whose content is a human-readable
    ``key: value`` representation of that row.  The CSV header is used as
    column names; if no header is detected the columns are numbered.

    The row number (1-based, excluding header) is stored in metadata so
    retrieved chunks can be traced back to the original spreadsheet row.
    """
    documents: List[Document] = []

    # Detect file encoding
    detected_encoding = _detect_encoding(file_path)

    with open(file_path, "r", encoding=detected_encoding, newline="") as fh:
        # Sniff dialect to handle comma/semicolon/tab-separated files
        sample = fh.read(4096)
        fh.seek(0)

        # Handle empty files
        if not sample.strip():
            raise ValueError("CSV file is empty. Please upload a CSV with at least one data row.")

        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # default fallback

        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(fh, dialect)

        if has_header:
            headers = next(reader)
            row_offset = 1
        else:
            # Peek at first row to determine column count
            try:
                first_row = next(reader)
            except StopIteration:
                raise ValueError("CSV file has no data rows. Please ensure the CSV contains at least one data row.")
            headers = [f"col_{i+1}" for i in range(len(first_row))]
            # Re-open would be complex; just process first_row manually
            _append_row_doc(documents, headers, first_row, 1, file_name)
            row_offset = 1

        for row_num, row in enumerate(reader, start=row_offset + 1):
            if not any(cell.strip() for cell in row):
                continue  # skip blank rows
            _append_row_doc(documents, headers, row, row_num, file_name)

        # Check if any documents were created (covers case of header-only CSV)
        if not documents:
            raise ValueError("CSV file contains only a header row with no data. Please ensure the CSV contains at least one data row.")

    logger.info("Loaded %d row(s) from CSV '%s'", len(documents), file_name)
    return documents


def _append_row_doc(
    documents: List[Document],
    headers: List[str],
    row: List[str],
    row_num: int,
    file_name: str,
) -> None:
    """Convert a single CSV row into a Document and append it to the list."""
    pairs = []
    for header, value in zip(headers, row):
        pairs.append(f"{header}: {value}")
    # Include any extra columns beyond the header count
    for extra_idx, value in enumerate(row[len(headers):], start=len(headers) + 1):
        pairs.append(f"col_{extra_idx}: {value}")

    content = "\n".join(pairs)
    if content.strip():
        documents.append(
            Document(
                page_content=content,
                metadata={"source": file_name, "row": row_num, "type": "csv"},
            )
        )
