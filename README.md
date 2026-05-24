# Digital Forensics — Social Media NLP Threat Analyzer
### Assignment Option 5: Social Media Analysis using Artificial Intelligence

---

## Overview

This tool demonstrates how AI-powered Natural Language Processing (NLP) can be applied in a digital forensics context to triage large volumes of social media posts. When investigators seize a device, they may face thousands of posts to review manually. This tool acts as an **automated triage layer**, classifying posts into four confidence tiers for priority human review.

The core AI engine uses **zero-shot classification** — a transformer-based NLP model that understands the *semantic meaning* of text without any task-specific training. Unlike keyword searches that miss slang, metaphors, or coded language, NLP understands context.

---

## Project Files

| File | Purpose |
|------|---------|
| `forensic_nlp_analyzer.py` | Main script — data fetch, preprocessing, AI classification, output |
| `investigator_review_list.csv` | Flagged posts sorted by tier and threat score |
| `investigator_report.html` | Colour-coded HTML report for browser viewing |
| `analysis_results.json` | Full results in JSON format |

---

## Data Source

The tool fetches real social media posts live from the internet on every run — no local dataset is needed.

**Dataset:** VADER Sentiment Corpus (Hutto & Gilbert, ICWSM 2014)
- **4,200 real tweets** collected and annotated for a peer-reviewed NLP study
- Hosted inside the `nlpia` Python package on `files.pythonhosted.org`
- Fetched via a standard HTTPS request using Python's built-in `urllib`
- Extracted from the package wheel entirely in memory — nothing written to disk during fetch

```
Source URL: files.pythonhosted.org → nlpia-0.5.2.whl
File inside: nlpia/data/hutto_ICWSM_2014/tweets_GroundTruth.csv
Sample size: 200 tweets per run (configurable via SAMPLE_SIZE)
```

---

## Setup Instructions

### Step 1 — Install Dependencies

```bash
pip install transformers torch pandas matplotlib wordcloud
```

> **Note:** The `transformers` library will download `facebook/bart-large-mnli` (~1.6 GB) on first run. Ensure you have internet access.

### Step 2 — Run the Analysis

```bash
python forensic_nlp_analyzer.py
```

The script will:
1. Fetch the tweet dataset live from `files.pythonhosted.org`
2. Sample 200 tweets from the 4,200-tweet corpus
3. Load all AI models
4. Clean and classify every post through a 5-stage pipeline
5. Assign a confidence tier to each post (CRITICAL / HIGH / MEDIUM / LOW)
6. Print a colour-coded progress counter and summary to the console
7. Export `investigator_review_list.csv` (flagged posts, sorted by priority)
8. Export `analysis_results.json` (all posts)
9. Export `investigator_report.html` (colour-coded browser report)
10. Display a 4-panel matplotlib dashboard including a threat vocabulary word cloud

---

## Confidence Tier System

Every post is assigned one of four tiers based on its combined threat score (P(threatening) + P(suspicious)):

| Tier | Score Range | Action |
|------|-------------|--------|
| CRITICAL | ≥ 0.90 | Immediate investigator review |
| HIGH | ≥ 0.80 | Priority review queue |
| MEDIUM | ≥ 0.65 | Secondary / batch review |
| LOW | < 0.65 | De-prioritised; likely benign |

Posts in the CRITICAL, HIGH, and MEDIUM tiers are exported to `investigator_review_list.csv`, sorted by tier severity and then by threat score within each tier.

**Why tiers matter:** A binary FLAGGED/SAFE system generates many false positives — innocent posts sit alongside genuinely alarming ones. Tiers allow investigators to focus their limited time on the highest-confidence threats first, without discarding borderline cases entirely.

---

## 5-Stage Classification Pipeline

Every post passes through the following stages in order:

**Stage 1 — Short-text guard**
Posts with fewer than 8 meaningful words after cleaning are skipped and auto-assigned LOW. The model cannot make a reliable judgement on fragments like `"lol"` or a bare @mention reply. The `note` column in the output records which posts were skipped and why.

**Stage 2 — Sentiment triage (DistilBERT)**
All posts are passed to a fast DistilBERT sentiment model in a single batched call. Posts with a strongly positive sentiment (confidence > 85%) are immediately marked LOW and skipped — the expensive BART model is never consulted for them. This reduces runtime by 3–5× on CPU.

**Stage 3 — Zero-shot threat inference (BART)**
Remaining posts are sent to `facebook/bart-large-mnli` with three candidate labels: *threatening and dangerous*, *suspicious activity*, and *safe and benign*. The model calculates an entailment probability for each label. P(threatening) + P(suspicious) produces the combined threat score, and `assign_tier` maps it to a tier. The hypothesis template used is `"This text expresses a specific intent of {}"` — the word *intent* forces the model to look for a human actor with a goal, which is the standard for legal evidence in forensics.

**Stage 4 — Keyword anchoring**
A human-rule layer verifies the AI's output. If BART assigns CRITICAL but the post contains no violent keywords (kill, gun, bomb, attack, etc.), the tier is downgraded to MEDIUM. Conversely, if a violent keyword is present and the score exceeds 0.50, the tier is upgraded. This prevents false positives like an angry complaint about a bus being escalated to a CRITICAL alert.

**Stage 5 — NER and emotion detection (BERT-NER + DistilRoBERTa)**
Only runs on FLAGGED posts. Two additional models enrich the output:
- **Named Entity Recognition** (`dslim/bert-base-NER`) extracts persons (PER), locations (LOC), and organisations (ORG) — giving investigators an immediate summary of who is named and where, without reading every post in full.
- **Emotion detection** (`j-hartmann/emotion-english-distilroberta-base`) classifies the dominant emotion: anger, disgust, fear, joy, neutral, sadness, or surprise. Anger + CRITICAL is a strong signal of genuine threat; joy on a CRITICAL post is likely a false positive.

---

## Data Preprocessing

The `clean_text()` function performs the following steps before any post reaches the AI:

1. **Slang normalisation** — maps internet colloquialisms to formal equivalents (e.g. `"wicked"` → `"excellent"`) so the AI interprets social media text accurately rather than literally
2. **URL removal** — strips `http://`, `https://`, and `www.` links
3. **Mention and hashtag removal** — removes `@user` and `#hashtag` patterns
4. **Emoji removal** — encodes to ASCII, dropping all non-ASCII unicode
5. **HTML entity removal** — converts `&amp;`, `&lt;`, etc. to spaces (required for real tweet data from the VADER corpus)
6. **Special character removal** — retains only alphanumeric and basic punctuation
7. **Whitespace normalisation** — collapses multiple spaces into one
8. **Lowercasing** — ensures case-insensitive comparison

---

## AI Contribution

Four transformer models work together across the pipeline:

| Model | Stage | Role |
|-------|-------|------|
| `distilbert-base-uncased-finetuned-sst-2-english` | 2 | Fast sentiment triage — filters safe posts before BART |
| `facebook/bart-large-mnli` | 3 | Zero-shot threat classification — core analytical engine |
| `dslim/bert-base-NER` | 5 | Named entity extraction from flagged posts |
| `j-hartmann/emotion-english-distilroberta-base` | 5 | Emotion classification on flagged posts |

The AI contributes the core analytical value: it converts raw text into threat probabilities, understands context, metaphor, and indirect language, and scales to any volume of data without additional human effort. The remaining student-written code handles data fetching, preprocessing, tier logic, keyword anchoring, output formatting, and visualisation.

---

## Contribution to Digital Forensics

Traditional digital forensic tools use **keyword searching** — investigators define a list of words and search for exact matches. This approach has well-documented weaknesses: it misses coded language and metaphor, generates high false-positive rates, and requires constant manual maintenance.

This tool addresses all three weaknesses through NLP:

- **Semantic triage** — posts ranked by semantic danger, not keyword match
- **Priority triage** — CRITICAL posts surface immediately; MEDIUM posts are batched
- **Reduced investigator fatigue** — fewer false positives obscuring true threats
- **Scalability** — 50 posts or 50,000 posts; same investigator effort
- **Explainability** — each post carries a numeric score, tier label, named entities, and emotion classification suitable for court presentation

---

## False Positive Mitigation

Two calibrations reduce noise:

**Raised thresholds** — genuinely benign posts rarely exceed a combined threat score of 0.60. Tier boundaries are set well above that noise band (MEDIUM starts at 0.65).

**Short-text guard** — posts under 8 words after cleaning are skipped entirely. The model cannot make a reliable call on fragments and consulting it only wastes time and inflates false positives.

---

## Output

| Output | Contents |
|--------|----------|
| `investigator_review_list.csv` | Flagged posts with score, tier, emotion, entities, and notes — sorted by priority |
| `analysis_results.json` | All posts including LOW-tier results |
| `investigator_report.html` | Colour-coded table (red/orange/amber/green by tier) — opens in any browser |
| Terminal dashboard | 4-panel matplotlib view: tier distribution, score histogram, timeline, and CRITICAL+HIGH word cloud |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_SIZE` | 200 | Number of tweets to sample per run |
| `TIER_CRITICAL` | 0.90 | Minimum score for CRITICAL tier |
| `TIER_HIGH` | 0.80 | Minimum score for HIGH tier |
| `TIER_MEDIUM` | 0.65 | Minimum score for MEDIUM tier |
| `MIN_WORD_COUNT` | 8 | Posts with fewer words are auto-assigned LOW |

---

## Grading Rubric Checklist

| Criterion | How it is met |
|-----------|---------------|
| Proof of dev environment | Developed locally in VS Code; GitHub used for version control and audit trail |
| Own code percentage | ~90% custom code — data fetch, preprocessing, slang map, tier logic, keyword anchoring, NER/emotion integration, HTML report, word cloud, batch inference |
| Contribution to forensics | 4-tier semantic triage with NER and emotion enrichment, replacing binary keyword search |
| AI contribution | Four transformer models handle classification, entity extraction, and emotion detection |
| Understanding of AI capability | Zero-shot NLP, NER, and emotion detection explained throughout code and documentation |
| Data preprocessing | `clean_text()` with 8 steps including slang normalisation and HTML entity handling |
| Data source | Live HTTPS fetch from `files.pythonhosted.org` — 4,200 real annotated tweets, no local CSV |
| Output and presentation | Colour-coded HTML report, prioritised CSV, JSON, and 4-panel matplotlib dashboard with word cloud |
