"""One-off helper: fetch SAT QB correct answers for Writing Questions PDF IDs."""
from __future__ import annotations

import re
import urllib.request

IDS = [
    "2b08f514",
    "afec1a70",
    "4d2736f0",
    "d3b7d7a3",
    "39ccb463",
    "e3edc138",
    "00221c00",
    "16631d34",
    "1d79a59d",
    "42e6cc83",
    "83898524",
    "fba5d8d1",
    "dc645172",
    "886dc9f9",
    "59a246dc",
    "e060dd6b",
    "6e071432",
    "6ea8c23f",
    "aab74a3b",
    "512f0ac9",
]


def fetch_answer(qid: str) -> str | None:
    url = (
        "https://sat-questions.onrender.com/question/"
        f"module:english-group:all-skill:all-difficulty:all-active:all/{qid}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", "replace")
    m = re.search(
        r"Correct Answer:\s*</span>\s*<span>\s*([A-D])\s*</span>",
        html,
        re.I,
    )
    if not m:
        m = re.search(
            r"Correct Answer:\s*</span>\s*([A-D])",
            html,
            re.I,
        )
    if not m:
        m = re.search(r"Correct Answer:\s*([A-D])", html, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    if m:
        blob = m.group(1)
        for pat in (
            r'"correctAnswer"\s*:\s*"([A-D])"',
            r'"correct_answer"\s*:\s*"([A-D])"',
            r'"letter"\s*:\s*"([A-D])"[^}]*"isCorrect"\s*:\s*true',
        ):
            hit = re.search(pat, blob, re.I)
            if hit:
                return hit.group(1).upper()
    return None


if __name__ == "__main__":
    for qid in IDS:
        try:
            ans = fetch_answer(qid)
            print(qid, ans or "?")
        except Exception as exc:
            print(qid, "ERR", exc)
