# Deploying AdaptivPrep to Streamlit Community Cloud

Manual steps — the deploy itself happens in the Streamlit Cloud UI and takes
about 10 minutes. The repo is deploy-ready; nothing below requires code changes.

## Prerequisites

- This repo on your GitHub account (`urrra39/AdaptivPrep`).
- A Streamlit Community Cloud account: <https://share.streamlit.io> → sign in with GitHub.
- An Anthropic API key (<https://console.anthropic.com>) — optional; without it
  the app runs with AI feedback disabled (a graceful Uzbek caption, not an error).

## Deploy the quiz app

1. share.streamlit.io → **New app**.
2. Repository: `urrra39/AdaptivPrep` · Branch: `main` · Main file: `src/app/quiz_app.py`.
3. Pick an app URL (e.g. `adaptivprep`).
4. **Advanced settings** → Python version: **3.11** (matches the CI matrix).
5. **Secrets — optional.** You do **not** need to provide an API key: each
   visitor pastes their own Anthropic key in the app's sidebar, and it is used
   only for their session (never stored or logged). To offer a default key for
   all visitors at your expense instead, paste this TOML in the Secrets UI
   (quotes required): `ANTHROPIC_API_KEY = "sk-ant-your-real-key"` — the app
   bridges `st.secrets` into the environment itself.
6. **Deploy**. First build takes several minutes (the requirements include
   `torch`, reserved for the Phase-10 DKT model).

## Deploy the dashboard (optional, second app)

Repeat with Main file: `src/app/dashboard.py`. No secrets needed.

> **Caveat:** two Cloud apps have **separate filesystems** — a separately
> deployed dashboard cannot see the quiz app's SQLite file, so it will show
> an empty user list. For a shared-storage demo, deploy only the quiz, or
> convert the dashboard to a Streamlit multipage app (`pages/` directory)
> so both run in one container.

## Verify after deploy

- Quiz loads the Uzbek login screen; answering works end-to-end.
- New username → prompted to **set a 4-digit PIN**; returning to that username
  requires it (wrong PIN → Uzbek error, no login). This is collision prevention
  for a shared demo, not full authentication.
- Dashboard: selecting a username also requires that user's PIN before any
  charts render; a PIN-less (legacy) account shows a "cannot be viewed" message.
- Blind test mode: answers advance silently (no correct/incorrect indicator);
  a live timer runs at the top; "Sessiyani yakunlash" opens the summary report.
- On the summary report, each mistake has an "AI izohi" button: **with** a key
  (sidebar or secret) it returns a Claude explanation; **without** any key it
  shows the static Uzbek fallback (expected degradation, not an error).

## Operations

- **Updates:** `git push` to `main` auto-redeploys.
- **Storage is ephemeral:** the SQLite response log resets on every
  redeploy/reboot. Acceptable for a demo; for persistence set the
  `ADAPTIVPREP_DB` env var to a mounted path, or swap Postgres in behind
  `src/data/schema.py`.
- **Logs / restart:** app menu (⋮) → *Manage app* → logs; *Reboot* clears state.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build hits resource limits installing `torch` | Temporarily remove `torch` from `requirements.txt` — nothing running imports it until Phase 10. |
| "Oh no." crash page | Manage app → read the traceback in logs; usually a secrets TOML typo (missing quotes). |
| AI feedback silent with a valid key | Confirm the secret name is exactly `ANTHROPIC_API_KEY`; reboot the app after editing secrets. |
