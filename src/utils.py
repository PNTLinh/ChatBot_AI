import re

def chunk_text(text: str, max_chars: int = 1000):
    """
    Tách văn bản dài thành các đoạn nhỏ < max_chars.
    """
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 1 > max_chars:
            chunks.append(current)
            current = p
        else:
            current += ("\n" + p if current else p)
    if current:
        chunks.append(current)
    return chunks

def sanitize_filename(name: str):
    """
    Loại bỏ ký tự đặc biệt để dùng làm tên file.
    """
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:50]
