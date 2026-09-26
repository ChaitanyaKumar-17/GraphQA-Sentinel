"""
ingestion/chunk.py

Splits scraped page content into overlapping chunks sized for retrieval,
preserving section-level metadata (url, page title, heading) so the
Generator node can cite sources accurately.
"""

import json
import re
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "data" / "raw" / "scraped_docs.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "processed" / "chunks.json"

TARGET_WORDS = 450
OVERLAP_WORDS = 60

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


def split_into_sections(markdown_text: str) -> list[dict]:
    """Splits on Markdown '#' headers (trafilatura's markdown output)."""
    lines = markdown_text.split("\n")

    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                })
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip(),
        })

    return sections


def chunk_text(text: str, target_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + target_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    chunk_id = 0

    for page in pages:
        for section in split_into_sections(page["raw_text"]):
            if len(section["text"].split()) < 20:
                continue
            for piece in chunk_text(section["text"], TARGET_WORDS, OVERLAP_WORDS):
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_id:05d}",
                    "url": page["url"],
                    "title": page["title"],
                    "section": section["heading"],
                    "text": piece,
                })
                chunk_id += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()