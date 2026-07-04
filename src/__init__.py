"""AdaptivPrep — AI-based personalized learning system.

A three-layer adaptive tutor:

  1. Student Modeling Layer  — Bayesian Knowledge Tracing (per-skill mastery).
  2. Recommendation Layer    — multi-armed bandit (what to study next).
  3. AI Feedback Layer       — LLM-generated, mistake-specific explanations.

The architecture is domain-agnostic; the initial domain is IELTS
vocabulary & grammar, defined entirely by data/skills.json and
data/questions.json.
"""

__version__ = "0.1.0"
