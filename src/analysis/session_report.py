"""Post-session analytics: per-bucket accuracy, weaknesses, and IELTS band estimates.

Speaking and Writing are excluded by design — only Reading, Grammar (language use), and
Vocabulary (lexical resource) bands are estimated from this session's data.

Accuracy is calculated against the FULL section quota (unanswered = wrong) so that
a user who quits early gets a realistic low score, not an inflated percentage.
"""
from __future__ import annotations

DEFAULT_QUOTAS = {"Reading": 50, "Grammar": 50, "Vocabulary": 50}


def accuracy_to_band(accuracy: float) -> float | None:
    """Map accuracy to an IELTS band.

    Returns ``None`` ONLY when accuracy is exactly 0 (zero correct answers).
    Any non-zero accuracy gets a band — even 1 correct out of 40 gets 1.0.
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
    if accuracy >= 0.12:
        return 2.5
    if accuracy >= 0.07:
        return 2.0
    if accuracy >= 0.03:
        return 1.5
    return 1.0


def build_session_report(
    bucket_stats: dict,
    skill_stats: dict,
    mastery: dict,
    mistakes: list,
    elapsed_secs: int,
    total: int,
    correct: int,
    *,
    quotas: dict | None = None,
) -> dict:
    """Return a structured report dict for the summary screen.

    ``quotas`` maps bucket names to the full section size (e.g. Reading=50).
    Accuracy is computed as correct / quota so unanswered questions count as wrong.
    """
    quotas = quotas or DEFAULT_QUOTAS
    total_quota = sum(quotas.get(b, 0) for b in ("Reading", "Grammar", "Vocabulary"))

    bucket_bands: dict[str, float | None] = {}
    bucket_accuracy: dict[str, float] = {}
    for bucket in ("Reading", "Grammar", "Vocabulary"):
        stats = bucket_stats.get(bucket) or {"correct": 0, "total": 0}
        bucket_correct = stats.get("correct", 0)
        quota = quotas.get(bucket, stats.get("total", 0))
        if quota <= 0:
            quota = stats.get("total", 0)
        acc = (bucket_correct / quota) if quota else 0.0
        bucket_accuracy[bucket] = acc
        if bucket_correct == 0:
            bucket_bands[bucket] = None
        else:
            bucket_bands[bucket] = accuracy_to_band(acc)

    scored = [b for b, v in bucket_bands.items() if v is not None]
    overall_band = (
        round(sum(bucket_bands[b] for b in scored) / len(scored), 1) if scored else None
    )

    overall_accuracy = (correct / total_quota) if total_quota else 0.0

    bucket_labels = {
        "Reading": "READING",
        "Grammar": "GRAMMAR",
        "Vocabulary": "VOCABULARY",
    }
    weaknesses = []
    for bucket in ("Reading", "Grammar", "Vocabulary"):
        stats = bucket_stats.get(bucket) or {"correct": 0, "total": 0}
        n_answered = stats.get("total", 0)
        bucket_correct = stats.get("correct", 0)
        quota = quotas.get(bucket, n_answered)
        if quota <= 0:
            continue
        acc = bucket_correct / quota
        weaknesses.append(
            {
                "skill_id": bucket,
                "name": bucket_labels.get(bucket, bucket),
                "category": bucket,
                "accuracy": acc,
                "correct": bucket_correct,
                "answered": n_answered,
                "quota": quota,
                "wrong": n_answered - bucket_correct,
                "unanswered": quota - n_answered,
                "mastery": 0.0,
            }
        )
    weaknesses.sort(key=lambda w: w["accuracy"])

    recommendations = _recommendations(weaknesses[:5], bucket_accuracy)

    return {
        "elapsed_secs": elapsed_secs,
        "total": total,
        "total_quota": total_quota,
        "correct": correct,
        "wrong": total - correct,
        "unanswered": total_quota - total,
        "accuracy": overall_accuracy,
        "bucket_accuracy": bucket_accuracy,
        "bucket_bands": bucket_bands,
        "overall_band": overall_band,
        "weaknesses": weaknesses[:8],
        "recommendations": recommendations,
        "mistake_count": len(mistakes),
        "quotas": quotas,
    }


def _recommendations(top_weak: list, bucket_accuracy: dict) -> list[str]:
    out: list[str] = []
    for w in top_weak:
        if w["accuracy"] < 0.6:
            out.append(
                f"{w['name']} bo'yicha ko'proq mashq qiling "
                f"(to'g'ri: {w['correct']}/{w['quota']}, aniqlik {100 * w['accuracy']:.0f}%)."
            )
    for bucket, acc in bucket_accuracy.items():
        if acc < 0.55:
            label = {"Reading": "READING", "Grammar": "GRAMMAR", "Vocabulary": "VOCABULARY"}.get(
                bucket, bucket
            )
            out.append(f"{label} bo'limi zaif — har kuni 15 daqiqa {label.lower()} mashqi qiling.")
    if not out:
        out.append("Yaxshi natija! Xatolarni tahlil qilib, zaif mavzularda mustahkamlang.")
    return out[:5]
