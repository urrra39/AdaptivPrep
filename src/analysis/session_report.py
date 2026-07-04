"""Post-session analytics: per-bucket accuracy, weaknesses, and IELTS band estimates.

Listening and Speaking are excluded by design — only Reading, Grammar (language use), and
Vocabulary (lexical resource) bands are estimated from this session's data.
"""
from __future__ import annotations


def accuracy_to_band(accuracy: float) -> float | None:
    """Map a (0, 1] accuracy to an IELTS band in [2.5, 9.0] (0.5 steps).

    Returns ``None`` when there are no correct answers (0% accuracy).
    """
    if accuracy <= 0.0:
        return None
    if accuracy >= 0.97:
        return 9.0
    if accuracy >= 0.93:
        return 8.5
    if accuracy >= 0.88:
        return 8.0
    if accuracy >= 0.82:
        return 7.5
    if accuracy >= 0.75:
        return 7.0
    if accuracy >= 0.68:
        return 6.5
    if accuracy >= 0.60:
        return 6.0
    if accuracy >= 0.52:
        return 5.5
    if accuracy >= 0.44:
        return 5.0
    if accuracy >= 0.36:
        return 4.5
    if accuracy >= 0.30:
        return 4.0
    if accuracy >= 0.24:
        return 3.5
    if accuracy >= 0.18:
        return 3.0
    if accuracy >= 0.10:
        return 2.5
    return None


def build_session_report(
    bucket_stats: dict,
    skill_stats: dict,
    mastery: dict,
    mistakes: list,
    elapsed_secs: int,
    total: int,
    correct: int,
) -> dict:
    """Return a structured report dict for the summary screen."""
    bucket_bands = {}
    bucket_accuracy = {}
    for bucket, stats in bucket_stats.items():
        n = stats.get("total", 0)
        bucket_correct = stats.get("correct", 0)
        acc = (bucket_correct / n) if n else 0.0
        bucket_accuracy[bucket] = acc
        if n == 0 or bucket_correct == 0:
            bucket_bands[bucket] = None
        else:
            bucket_bands[bucket] = accuracy_to_band(acc)

    scored = [b for b, v in bucket_bands.items() if v is not None]
    overall_band = round(sum(bucket_bands[b] for b in scored) / len(scored), 1) if scored else None

    bucket_labels = {
        "Reading": "READING",
        "Grammar": "Grammatika",
        "Vocabulary": "Lug'at",
    }
    weaknesses = []
    for bucket in ("Reading", "Grammar", "Vocabulary"):
        stats = bucket_stats.get(bucket) or {"correct": 0, "total": 0}
        n = stats.get("total", 0)
        if n == 0:
            continue
        acc = stats["correct"] / n
        weaknesses.append(
            {
                "skill_id": bucket,
                "name": bucket_labels.get(bucket, bucket),
                "category": bucket,
                "accuracy": acc,
                "total": n,
                "mastery": 0.0,
            }
        )
    weaknesses.sort(key=lambda w: w["accuracy"])

    recommendations = _recommendations(weaknesses[:5], bucket_accuracy)

    return {
        "elapsed_secs": elapsed_secs,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": (correct / total) if total else 0.0,
        "bucket_accuracy": bucket_accuracy,
        "bucket_bands": bucket_bands,
        "overall_band": overall_band,
        "weaknesses": weaknesses[:8],
        "recommendations": recommendations,
        "mistake_count": len(mistakes),
    }


def _recommendations(top_weak: list, bucket_accuracy: dict) -> list[str]:
    out = []
    for w in top_weak:
        if w["accuracy"] < 0.6:
            out.append(
                f"{w['name']} bo'yicha ko'proq mashq qiling "
                f"(sessiya aniqligi {100 * w['accuracy']:.0f}%)."
            )
    for bucket, acc in bucket_accuracy.items():
        if acc < 0.55:
            label = {"Reading": "READING", "Grammar": "Grammatika", "Vocabulary": "Lug'at"}.get(
                bucket, bucket
            )
            out.append(f"{label} bo'limi zaif — har kuni 15 daqiqa {label.lower()} mashqi qiling.")
    if not out:
        out.append("Yaxshi natija! Xatolarni tahlil qilib, zaif mavzularda mustahkamlang.")
    return out[:5]
