import json
from pathlib import Path

def stream_arxiv_json(path):
    """
    Generator that yields one paper at a time
    without loading the full file into memory.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue 