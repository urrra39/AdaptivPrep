"""Persistent session history helpers: compare runs, charts, motivational copy."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import plotly.graph_objects as go

TASHKENT_TZ = timezone(timedelta(hours=5), name="UZB")

PHASE_LABELS = {
    "Reading": "O'qish",
    "Grammar": "Grammatika",
    "Vocabulary": "Lug'at",
}


def format_duration(secs: int) -> str:
    mins, s = divmod(max(0, int(secs)), 60)
    return f"{mins:02d}:{s:02d}"


def format_completed_at(iso_utc: str) -> str:
    """Render a stored UTC ISO timestamp in Uzbekistan local time (Asia/Tashkent)."""
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TASHKENT_TZ).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_utc


def compare_sessions(older: dict, newer: dict) -> dict:
    """Compare two saved session rows (older first, newer second)."""
    ob = older.get("overall_band")
    nb = newer.get("overall_band")
    oa = float(older.get("accuracy") or 0.0)
    na = float(newer.get("accuracy") or 0.0)
    band_delta = (nb - ob) if ob is not None and nb is not None else None
    acc_delta = na - oa
    ratio = (nb / ob) if ob and nb and ob > 0 else None
    improved = (band_delta is not None and band_delta > 0) or acc_delta > 0.05
    return {
        "older_band": ob,
        "newer_band": nb,
        "band_delta": band_delta,
        "accuracy_delta": acc_delta,
        "band_ratio": ratio,
        "improved": improved,
        "message": motivation_message(band_delta, acc_delta, improved),
    }


def motivation_message(
    band_delta: float | None, acc_delta: float, improved: bool
) -> str:
    if band_delta is not None and band_delta >= 1.0:
        return (
            f"Ajoyib! Taxminiy IELTS balingiz {band_delta:.1f} ga oshdi — "
            "Yuksalish bor! Davom eting!"
        )
    if band_delta is not None and band_delta > 0:
        return (
            f"Yuksalish bor! IELTS taxminiy bali +{band_delta:.1f}. "
            "Har bir sessiya sizni oldinga suryapti!"
        )
    if improved:
        return (
            "Yuksalish bor! Aniqligingiz yaxshilandi — "
            "shu tempo bilan davom eting!"
        )
    if band_delta is not None and band_delta < 0:
        return (
            "Bu sessiya oldingisidan past — xatolarni tahlil qilib "
            "keyingi urinishda yaxshilang."
        )
    return "Barqaror natija. Mashqni davom ettiring — har kuni oz-ozdan o'sasiz!"


def band_score_pie(row: dict, title: str, *, tone: str = "old", dark: bool = False) -> go.Figure:
    """Donut chart for IELTS band (0–9). tone: old (grey), good (green), bad (red)."""
    band = row.get("overall_band")
    # Old-money chart palette: parchment / racing green / antique gold.
    palettes = {
        "old": ["#C5A059", "#4A4436"],
        "good": ["#4E7A57", "#22392B"],
        "bad": ["#A9503C", "#4C2A20"],
    }
    colors = palettes.get(tone, palettes["old"])
    paper_bg = "#0B1610" if dark else "#FBF7EB"
    font_color = "#E6D7AE" if dark else "#1F3D2B"

    if band is None:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Natija yo'q"],
                    values=[1],
                    marker={"line": {"color": "#C5A059", "width": 1.2}, "colors": ["#8A7F63"]},
                    hole=0.55,
                    textinfo="label",
                    textfont={"size": 14, "color": font_color, "family": "Georgia, serif"},
                )
            ]
        )
        fig.add_annotation(
            text="<b>—</b>",
            x=0.5,
            y=0.5,
            font={"size": 28, "color": font_color, "family": "Georgia, serif"},
            showarrow=False,
        )
    else:
        band_f = float(band)
        rest = max(0.01, 9.0 - band_f)
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[f"Band {band_f:.1f}", ""],
                    values=[band_f, rest],
                    marker={"line": {"color": "#C5A059", "width": 1.2}, "colors": colors},
                    hole=0.55,
                    textinfo="none",
                    hoverinfo="skip",
                )
            ]
        )
        fig.add_annotation(
            text=f"<b>{band_f:.1f}</b>",
            x=0.5,
            y=0.5,
            font={"size": 30, "color": font_color, "family": "Georgia, serif"},
            showarrow=False,
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"color": font_color, "family": "Georgia, serif"}},
        margin={"t": 55, "b": 20, "l": 20, "r": 20},
        height=340,
        showlegend=False,
        font={"family": "Georgia, serif", "color": font_color},
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
    )
    return fig


def band_overview(rows: list[dict]) -> list[tuple[str, dict, str]]:
    """Specs for the four overview donuts: best / average / worst / current.

    Returns a list of (title, row, tone) tuples, newest session first in rows.
    Tones: best=good (green), average=old (neutral gold), worst=bad (copper),
    current is judged against the average. If every session has the same band,
    all four donuts stay neutral gold (no fake green/red).
    """
    bands = [
        float(r["overall_band"])
        for r in rows
        if r.get("overall_band") is not None
    ]
    current_band = rows[0].get("overall_band") if rows else None
    if not bands:
        none_row = {"overall_band": None}
        return [
            ("ENG YUQORI", dict(none_row), "old"),
            ("O'RTACHA", dict(none_row), "old"),
            ("ENG PASTI", dict(none_row), "old"),
            ("HOZIRGI", dict(none_row), "old"),
        ]
    best = max(bands)
    worst = min(bands)
    avg = round(sum(bands) / len(bands), 1)
    if best == worst:
        tones = ("old", "old", "old", "old")
    else:
        cur = float(current_band) if current_band is not None else None
        if cur is None or cur == avg:
            cur_tone = "old"
        elif cur > avg:
            cur_tone = "good"
        else:
            cur_tone = "bad"
        tones = ("good", "old", "bad", cur_tone)
    return [
        ("ENG YUQORI", {"overall_band": best}, tones[0]),
        ("O'RTACHA", {"overall_band": avg}, tones[1]),
        ("ENG PASTI", {"overall_band": worst}, tones[2]),
        ("HOZIRGI", {"overall_band": current_band}, tones[3]),
    ]


def accuracy_pie(row: dict, title: str) -> go.Figure:
    """Pie chart of correct vs wrong answers for one session."""
    correct = int(row.get("correct") or 0)
    wrong = int(row.get("wrong") or 0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["To'g'ri", "Xato"],
                values=[correct, wrong],
                marker={"line": {"color": "#C5A059", "width": 1.2}, "colors": ["#4E7A57", "#B0614B"]},
                hole=0.35,
                textinfo="label+percent",
                textfont={"size": 14},
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
        height=320,
        showlegend=True,
    )
    fig.update_layout(font={"family": "Georgia, serif"})
    return fig


def bucket_error_pie(row: dict, title: str) -> go.Figure:
    """Pie chart of wrong answers split by Reading / Grammar / Vocabulary."""
    stats = row.get("bucket_stats") or {}
    labels, values = [], []
    colors = {"Reading": "#C5A059", "Grammar": "#4E7A57", "Vocabulary": "#8C5A3C"}
    slice_colors = []
    for bucket in ("Reading", "Grammar", "Vocabulary"):
        block = stats.get(bucket) or {}
        total = int(block.get("total") or 0)
        correct = int(block.get("correct") or 0)
        wrong = max(0, total - correct)
        if wrong > 0:
            labels.append(PHASE_LABELS.get(bucket, bucket))
            values.append(wrong)
            slice_colors.append(colors[bucket])
    if not values:
        labels, values, slice_colors = ["Xato yo'q"], [1], ["#4E7A57"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker={"line": {"color": "#C5A059", "width": 1.2}, "colors": slice_colors},
                hole=0.35,
                textinfo="label+value",
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
        height=320,
    )
    fig.update_layout(font={"family": "Georgia, serif"})
    return fig


def top_weakness_lines(row: dict, limit: int = 5) -> list[str]:
    """Human-readable weak skills from a saved session row."""
    weak = row.get("weaknesses") or []
    lines = []
    for w in weak[:limit]:
        name = w.get("name") or w.get("skill_id") or "Noma'lum"
        acc = 100 * float(w.get("accuracy") or 0.0)
        # Weakness rows store "quota"/"answered" (see session_report); the old
        # "total" key never existed, so this always rendered "(0 savol)".
        count = int(w.get("quota") or w.get("answered") or w.get("total") or 0)
        lines.append(f"{name}: {acc:.0f}% aniqlik ({count} savol)")
    return lines


def build_results_ai_prompt(older: dict, newer: dict, cmp_: dict) -> str:
    """Prompt for AI coach to explain mistakes and improvement areas."""
    old_weak = top_weakness_lines(older) or ["Ma'lumot yo'q"]
    new_weak = top_weakness_lines(newer) or ["Ma'lumot yo'q"]
    return (
        "Ikki IELTS mashq sessiyasini solishtiring va o'zbek tilida tahlil bering.\n"
        f"OLDINGI: IELTS taxmin {older.get('overall_band')}, "
        f"to'g'ri {older.get('correct')}, xato {older.get('wrong')}, "
        f"aniqlik {100 * float(older.get('accuracy') or 0):.0f}%.\n"
        f"YANGI: IELTS taxmin {newer.get('overall_band')}, "
        f"to'g'ri {newer.get('correct')}, xato {newer.get('wrong')}, "
        f"aniqlik {100 * float(newer.get('accuracy') or 0):.0f}%.\n"
        f"O'zgarish: band delta {cmp_.get('band_delta')}, "
        f"aniqlik delta {100 * float(cmp_.get('accuracy_delta') or 0):+.0f}%.\n"
        f"Oldingi zaifliklar: {'; '.join(old_weak)}.\n"
        f"Yangi zaifliklar: {'; '.join(new_weak)}.\n"
        "Qaysi bo'limlarda (Reading/Grammar/Vocabulary) ko'proq xato bor, "
        "muammo nimada, va keyingi 3 ta aniq tavsiya bering."
    )
