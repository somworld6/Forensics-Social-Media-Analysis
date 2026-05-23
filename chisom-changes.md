# chisom-changes.md
## Enhancements to `forensic_nlp_analyzer.py`

---

### Enhancement 1 — Named Entity Recognition (NER) on flagged posts
**Forensic value: HIGH**

**What it does:**
After a post is classified CRITICAL or HIGH, a lightweight NER model (`dslim/bert-base-NER`) extracts:
- **Persons** (PER) — who is named?
- **Locations** (LOC) — where is this happening?
- **Organizations** (ORG) — any group named?

These entities are stored as an extra `entities` column in both the CSV and JSON outputs. Investigators get a fast-scan summary of names and places without reading every post in full.

**Why it matters:**
- Adds a third AI model (NER alongside DistilBERT sentiment and BART zero-shot)
- Directly demonstrates a real forensic use case — identifying suspects and locations
- Enhances "AI contribution" and "contribution to the field of digital forensics" rubric criteria

**New dependency:** None — `transformers` already supports NER pipelines.

---

### Enhancement 2 — Emotion detection on CRITICAL/HIGH posts
**Forensic value: MEDIUM-HIGH**

**What it does:**
Adds a fourth AI model — `j-hartmann/emotion-english-distilroberta-base` — which classifies text into one of seven emotions: anger, disgust, fear, joy, neutral, sadness, surprise. The dominant emotion is stored in a new `emotion` column for every flagged post.

This matters forensically because:
- **Anger + CRITICAL** → plausible genuine threat
- **Fear** → the author may be a victim, not a perpetrator
- **Joy** → likely false positive (boasting metaphorically)

**New dependency:** None — no extra install needed.

---

### Enhancement 3 — Progress counter during classification
**Forensic value: LOW (UX)**

**What it does:**
Prints `[  12 / 200]` at the start of processing each post so the terminal doesn't appear frozen during the ~2-minute BART inference run.

---

### Enhancement 4 — Colour-coded HTML investigator report
**Forensic value: MEDIUM**

**What it does:**
Generates `investigator_report.html` alongside the existing CSV. Each row is colour-coded by tier (red/orange/amber/green) and the table includes the original post, threat score, tier, emotion, NER entities, and analyst notes. Opens in any browser — far more readable than a raw CSV for a live demo or submission.

**Why it matters:**
- Directly addresses the "final output and presentation of results" rubric criterion (10 pts)
- No extra library needed — pure Python string templating

---

### Enhancement 5 — Word cloud panel for CRITICAL+HIGH posts
**Forensic value: MEDIUM**

**What it does:**
Replaces the scatter plot (Panel 4) in the dashboard with a `Reds` colourmap word cloud of the most frequent terms appearing in CRITICAL and HIGH posts. Gives a visual "threat vocabulary" snapshot — useful for spotting coordinated language or recurring themes across flagged content.

Falls back gracefully to the scatter plot if the `wordcloud` library is not installed.

**New dependency:** `pip install wordcloud`

---

### Enhancement 6 — Batch sentiment inference
**Forensic value: LOW (performance)**

**What it does:**
Previously `sentiment_pipe()` was called once per post inside the classification loop. All 200 texts are now passed to DistilBERT in a single batched call (`batch_size=32`), which is 3–5x faster on CPU because the tokenizer and model process them as a matrix rather than sequentially.

---

### Summary Table

| # | Enhancement | Rubric Impact |
|---|-------------|---------------|
| 1 | NER extracts persons, locations, orgs from flagged posts | AI capability, forensics contribution |
| 2 | Emotion detection on CRITICAL/HIGH posts | AI capability, AI contribution score |
| 3 | Progress counter `[X / 200]` in classification loop | Presentation |
| 4 | Colour-coded HTML investigator report | Final output & presentation (10 pts) |
| 5 | Word cloud panel for CRITICAL+HIGH vocabulary | Final output & presentation (10 pts) |
| 6 | Batch sentiment inference (faster runtime) | Efficiency |
