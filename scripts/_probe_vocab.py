"""Probe Vocabook Fighting Time layout."""
from __future__ import annotations

import re
import sys

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = r"C:\Users\Fayzulloh\Downloads\Vocabook.pdf"

with pdfplumber.open(PDF) as pdf:
    for i, page in enumerate(pdf.pages):
        t = page.extract_text() or ""
        if re.search(r"Fight(?:ing)? Time", t, re.I):
            qs = len(re.findall(r"^\d+\.\s+", t, re.M))
            print(f"page {i+1}: {qs} numbered items")
