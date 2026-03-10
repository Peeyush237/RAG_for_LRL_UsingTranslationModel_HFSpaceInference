"""
chunker.py
──────────
Sentence-aware text chunking with metadata.

Improvements over the original word-level splitter:
  • Splits on real sentence boundaries (English periods, Odia '।')
  • Never breaks mid-sentence
  • Smaller default chunk size (200 words) for better semantic focus
  • Each chunk carries source metadata for traceability
"""

import re
from dataclasses import dataclass, field

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 200       # words
DEFAULT_CHUNK_OVERLAP = 40     # words


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    index: int
    source: str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


# ── Sentence splitting ────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Handles:
      - English sentence endings (. ! ?)
      - Odia sentence endings (।)
      - Preserves abbreviations like "Dr.", "Mr.", "U.S.A."
    """
    # Split on sentence-ending punctuation followed by whitespace
    # Uses lookbehind to keep the punctuation with the sentence
    pattern = r'(?<=[।\.!\?])\s+'
    raw_sentences = re.split(pattern, text.strip())

    # Filter empty strings and very short fragments
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Merge very short fragments (< 5 words) with the previous sentence
    merged = []
    for sent in sentences:
        if merged and len(sent.split()) < 5:
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)

    return merged


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_words: int = 15,
) -> list[Chunk]:
    """
    Split text into overlapping, sentence-aware chunks.

    Algorithm:
      1. Split text into sentences
      2. Accumulate sentences into a chunk until word count >= chunk_size
      3. When a chunk is full, start the next chunk with `overlap` words
         of trailing context from the previous chunk
      4. Discard chunks shorter than min_chunk_words

    Args:
        text:            Raw text to chunk
        source:          Source filename for metadata
        chunk_size:      Target words per chunk
        overlap:         Words of overlap between consecutive chunks
        min_chunk_words: Minimum words for a chunk to be kept

    Returns:
        List of Chunk objects with text, index, source, and word_count
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        sent_words = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += sent_words

        # Check if we've reached the chunk size
        if current_word_count >= chunk_size:
            chunk_text_str = " ".join(current_sentences)

            chunks.append(Chunk(
                text=chunk_text_str,
                index=chunk_index,
                source=source,
            ))
            chunk_index += 1

            # Build overlap from trailing words of the current chunk
            all_words = chunk_text_str.split()
            if overlap > 0 and len(all_words) > overlap:
                overlap_text = " ".join(all_words[-overlap:])
                current_sentences = [overlap_text]
                current_word_count = overlap
            else:
                current_sentences = []
                current_word_count = 0

    # Handle remaining text
    if current_sentences:
        remainder = " ".join(current_sentences)
        if len(remainder.split()) >= min_chunk_words:
            chunks.append(Chunk(
                text=remainder,
                index=chunk_index,
                source=source,
            ))

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of {"text": str, "source": str} dicts
        chunk_size: Target words per chunk
        overlap: Words of overlap

    Returns:
        Flat list of all chunks across documents, re-indexed sequentially
    """
    all_chunks: list[Chunk] = []
    global_index = 0

    for doc in documents:
        doc_chunks = chunk_text(
            text=doc.get("text", ""),
            source=doc.get("source", "unknown"),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in doc_chunks:
            chunk.index = global_index
            all_chunks.append(chunk)
            global_index += 1

    return all_chunks
