"""Remove OCR-corrupted satoplam grammar items and top-up from the supplement pool."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = ROOT / "data" / "grammar_bank.json"

OPT_JUNK = [
    re.compile(r"[\^»«]"),
    re.compile(r"::+"),
    re.compile(r"[a-z]{2}i{3,}", re.I),
]


def is_bad(q: dict) -> bool:
    prompt = q.get("question_text", "") or ""
    if re.search(r"\bEnglish\?\s*[a-z]", prompt):
        return True
    for opt in q.get("options", []):
        text = opt or ""
        for pat in OPT_JUNK:
            if pat.search(text):
                return True
        if not text.strip():
            return True
    return False


def main() -> None:
    with open(BANK_PATH, encoding="utf-8") as fh:
        bank = json.load(fh)

    original = list(bank["questions"])
    kept = [q for q in original if not is_bad(q)]
    removed = len(original) - len(kept)

    from _add_grammar_supplement import NEW_QUESTIONS

    existing_ids = {q["id"] for q in kept}
    added = 0
    for template in NEW_QUESTIONS:
        if len(kept) >= 50:
            break
        seed = template["question_text"][:60]
        qid = "gram_" + hashlib.md5(seed.encode()).hexdigest()[:8]
        if qid in existing_ids:
            continue
        q = dict(template)
        q["id"] = qid
        q["external_id"] = qid.split("_", 1)[1]
        q["source"] = f"Grammar supplement ({q['skill_id']})"
        q["bank"] = "grammar"
        kept.append(q)
        existing_ids.add(qid)
        added += 1

    bank["questions"] = kept
    bank["meta"]["question_count"] = len(kept)
    bank["meta"]["note"] = (
        f"Cleaned SAT + supplement bank: removed {removed} OCR-corrupted rows, "
        f"topped up with {added} supplementary items. Total: {len(kept)}."
    )

    with open(BANK_PATH, "w", encoding="utf-8") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=2)

    print(f"Removed {removed} bad rows, added {added} new. Final total: {len(kept)}.")


if __name__ == "__main__":
    main()
