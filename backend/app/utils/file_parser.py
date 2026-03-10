"""
file_parser.py
──────────────
PDF and TXT file parsing — extracted from original rag.py.
No Streamlit dependency.
"""

import io


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def parse_txt(file_bytes: bytes) -> str:
    """Decode TXT file bytes to string."""
    return file_bytes.decode("utf-8", errors="ignore")


def parse_file(filename: str, file_bytes: bytes) -> str:
    """
    Parse a file by extension.

    Args:
        filename:   Original filename (used to detect type)
        file_bytes: Raw file content as bytes

    Returns:
        Extracted text string

    Raises:
        ValueError: If file type is not supported
    """
    name_lower = filename.lower()

    if name_lower.endswith(".txt"):
        return parse_txt(file_bytes)
    elif name_lower.endswith(".pdf"):
        return parse_pdf(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Only PDF and TXT are supported.")
