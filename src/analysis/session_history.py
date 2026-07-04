"""Persistent session history helpers: compare runs, charts, motivational copy."""
from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go

PHASE_LABELS = {
    "Reading": "O'qish",
    "Grammar": "Grammatika",
    "Vocabulary": "Lug'at",
}


def format_duration(secs: int) -> str:
    mins, s = divmod(max(0, int(secs)), 60)
    return f"{mins:02d}:{s:02d}"


def format_completed_at(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
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
    palettes = {
        "old": ["#9e9e9e", "#424242"],
        "good": ["#2ecc71", "#1b4332"],
        "bad": ["#e74c3c", "#641e1e"],
    }
    colors = palettes.get(tone, palettes["old"])
    paper_bg = "#000000" if dark else "#ffffff"
    font_color = "#ffffff" if dark else "#0a3d62"

    if band is None:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Natija yo'q"],
                    values=[1],
                    marker={"colors": ["#e74c3c"]},
                    hole=0.55,
                    textinfo="label",
                    textfont={"size": 14, "color": font_color},
                )
            ]
        )
        fig.add_annotation(
            text="<b>—</b>",
            x=0.5,
            y=0.5,
            font={"size": 28, "color": font_color},
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
                    marker={"colors": colors},
                    hole=0.55,
                    textinfo="none",
                    hovertemplate="IELTS band: %{value:.1f}<extra></extra>",
                )
            ]
        )
        fig.add_annotation(
            text=f"<b>{band_f:.1f}</b>",
            x=0.5,
            y=0.5,
            font={"size": 30, "color": font_color},
            showarrow=False,
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"color": font_color}},
        margin={"t": 55, "b": 20, "l": 20, "r": 20},
        height=340,
        showlegend=False,
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
    )
    return fig


def accuracy_pie(row: dict, title: str) -> go.Figure:
    """Pie chart of correct vs wrong answers for one session."""
    correct = int(row.get("correct") or 0)
    wrong = int(row.get("wrong") or 0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["To'g'ri", "Xato"],
                values=[correct, wrong],
                marker={"colors": ["#2ecc71", "#e74c3c"]},
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
    return fig


def bucket_error_pie(row: dict, title: str) -> go.Figure:
    """Pie chart of wrong answers split by Reading / Grammar / Vocabulary."""
    stats = row.get("bucket_stats") or {}
    labels, values = [], []
    colors = {"Reading": "#3498db", "Grammar": "#9b59b6", "Vocabulary": "#e67e22"}
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
        labels, values, slice_colors = ["Xato yo'q"], [1], ["#2ecc71"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker={"colors": slice_colors},
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
    return fig


def top_weakness_lines(row: dict, limit: int = 5) -> list[str]:
    """Human-readable weak skills from a saved session row."""
    weak = row.get("weaknesses") or []
    lines = []
    for w in weak[:limit]:
        name = w.get("name") or w.get("skill_id") or "Noma'lum"
        acc = 100 * float(w.get("accuracy") or 0.0)
        total = int(w.get("total") or 0)
        lines.append(f"{name}: {acc:.0f}% aniqlik ({total} savol)")
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
