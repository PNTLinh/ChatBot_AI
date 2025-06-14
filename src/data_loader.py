import json
from typing import List, Tuple, Dict

def load_procedures(path: str) -> Tuple[List[str], List[Dict]]:
    """
    Đọc file JSON chứa danh sách thủ tục, trả về:
      - docs: list các chuỗi văn bản (đã ghép title + content)
      - metas: list các dict metadata tương ứng (title, source, url,…)
    """
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    docs = []
    metas = []
    for item in items:
        title   = item.get("title", "").strip()
        content = item.get("content", "").strip()

<<<<<<< HEAD
=======
        # Ghép title và content để embed; nếu không cần có thể chỉ lấy content
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
        text = f"{title}\n\n{content}" if title else content
        docs.append(text)

        metas.append({
            "title":  title,
            "source": item.get("source", "").strip(),
            "url":    item.get("url", "").strip(),
        })

    return docs, metas
