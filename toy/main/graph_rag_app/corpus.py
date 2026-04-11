from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    return latin_tokens + cjk_chars


def load_docx_text(docx_path: str | Path) -> str:
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到文档：{path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"当前仅支持 .docx 文档：{path}")

    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)


def load_markdown_text(markdown_path: str | Path) -> str:
    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 Markdown 文件：{path}")
    if path.suffix.lower() != ".md":
        raise ValueError(f"当前仅支持 .md Markdown 文件：{path}")
    return path.read_text(encoding="utf-8").strip()


def load_markdown_directory_text(directory_path: str | Path) -> str:
    root = Path(directory_path)
    if not root.exists():
        raise FileNotFoundError(f"未找到知识库目录：{root}")
    if not root.is_dir():
        raise ValueError(f"知识库路径不是目录：{root}")

    markdown_files = sorted(path for path in root.rglob("*.md") if path.is_file())
    if not markdown_files:
        raise ValueError(f"No markdown files found in directory: {root}")

    blocks: list[str] = []
    for path in markdown_files:
        text = load_markdown_text(path)
        if not text:
            continue
        source = path.relative_to(root).as_posix()
        blocks.append(f"[SOURCE: {source}]\n\n{text}")

    if not blocks:
        raise ValueError(f"No non-empty markdown files found in directory: {root}")
    return "\n\n".join(blocks)


def load_corpus_text(kb_path: str | Path) -> str:
    path = Path(kb_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到知识库路径：{path}")

    if path.is_dir():
        return load_markdown_directory_text(path)

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return load_docx_text(path)
    if suffix == ".md":
        return f"[SOURCE: {path.name}]\n\n{load_markdown_text(path)}"

    raise ValueError(f"不支持的知识库路径类型：{path}")


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    sentences = [
        segment.strip()
        for segment in SENTENCE_BOUNDARY_PATTERN.split(stripped)
        if segment.strip()
    ]
    return sentences or [stripped]


def _tail_with_boundaries(text: str, limit: int) -> str:
    if limit <= 0:
        return ""

    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    selected: list[str] = []
    total = 0
    for paragraph in reversed(paragraphs):
        addition = len(paragraph) + (2 if selected else 0)
        if selected and total + addition > limit:
            break
        if len(paragraph) <= limit and total + addition <= limit:
            selected.append(paragraph)
            total += addition
            continue
        break
    if selected:
        return "\n\n".join(reversed(selected))

    sentences = _split_sentences(text)
    selected_sentences: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        addition = len(sentence) + (1 if selected_sentences else 0)
        if selected_sentences and total + addition > limit:
            break
        if len(sentence) <= limit and total + addition <= limit:
            selected_sentences.append(sentence)
            total += addition
            continue
        break
    if selected_sentences:
        return " ".join(reversed(selected_sentences))

    return text[-limit:].strip()


def _chunk_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        chunks: list[str] = []
        start = 0
        step = max(1, chunk_size - chunk_overlap)
        while start < len(text):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            start += step
        return chunks

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap_text = _tail_with_boundaries(current, chunk_overlap)
            current = f"{overlap_text} {sentence}".strip() if overlap_text else sentence
            if len(current) > chunk_size:
                chunks.extend(_chunk_long_text(current, chunk_size, chunk_overlap=0))
                current = ""
            continue

        chunks.extend(_chunk_long_text(sentence, chunk_size, chunk_overlap=0))
        current = ""

    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = _tail_with_boundaries(current, chunk_overlap) if chunk_overlap else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            chunks.extend(_chunk_long_text(paragraph, chunk_size, chunk_overlap))
            current = ""

    if current:
        chunks.append(current)

    return chunks
