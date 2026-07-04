"""DEEP FIX: physically sanitize question prompts + options in the JSON banks so
source names (ELS) and OCR junk can NEVER reach a user, regardless of display code,
deployment lag, or browser cache. Idempotent — safe to run repeatedly."""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data import loader

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def clean_text(t: str) -> str:
    return loader.clean_display_prompt(t or "")


def bake_reading(path: pathlib.Path) -> tuple[int, int]:
    bank = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for p in bank["passages"]:
        for q in p.get("questions", []):
            orig = q.get("question_text", "")
            new = clean_text(orig)
            if new != orig:
                q["question_text"] = new
                changed += 1
            opts = q.get("options")
            if isinstance(opts, list):
                cleaned_opts = [clean_text(o) if isinstance(o, str) else o for o in opts]
                if cleaned_opts != opts:
                    q["options"] = cleaned_opts
                    changed += 1
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(p.get("questions", [])) for p in bank["passages"])
    return changed, total


def bake_flat(path: pathlib.Path) -> tuple[int, int]:
    bank = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for q in bank["questions"]:
        orig = q.get("question_text", "")
        new = clean_text(orig)
        if new != orig:
            q["question_text"] = new
            changed += 1
        opts = q.get("options")
        if isinstance(opts, list):
            cleaned_opts = [clean_text(o) if isinstance(o, str) else o for o in opts]
            if cleaned_opts != opts:
                q["options"] = cleaned_opts
                changed += 1
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed, len(bank["questions"])


if __name__ == "__main__":
    c, t = bake_reading(DATA / "reading_bank.json")
    print(f"reading   : cleaned {c} fields across {t} questions")
    c, t = bake_flat(DATA / "grammar_bank.json")
    print(f"grammar   : cleaned {c} fields across {t} questions")
    c, t = bake_flat(DATA / "vocabulary_bank.json")
    print(f"vocabulary: cleaned {c} fields across {t} questions")
