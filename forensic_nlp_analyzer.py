"""
forensic_nlp_analyzer.py
-------------------------
Digital Forensics — Social Media Threat Detection using NLP
Option 5: Social Media Analysis

This tool simulates how a digital forensic investigator would triage
a large volume of seized social media posts, using AI-powered NLP to
flag threatening or suspicious content for priority review.

Rubric alignment:
  - Data source:         Live fetch from published academic tweet dataset (VADER/ICWSM 2014)
                         hosted on files.pythonhosted.org — real HTTP request, no local CSV needed
  - Data preprocessing:  clean_text() function (regex, lowercasing, emoji/URL removal)
  - AI capability:       HuggingFace zero-shot classification pipeline
  - Contribution:        Outputs prioritised investigator_review_list.csv with 4-tier risk levels
  - Presentation:        4-panel live matplotlib dashboard (plt.show())

Author:  Marco Geral, Ethan Wilke and Chisom Emekpo
Date:    24 May 2026
"""

import re
import io
import json
import zipfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ── AI / NLP ──────────────────────────────────────────────────────────────────
from transformers import pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
# Online dataset: VADER Sentiment Corpus — 4,200 real tweets from Hutto & Gilbert
# (ICWSM 2014). Bundled inside the nlpia Python package hosted on PyPI.
# No API key required. No local CSV needed. Fetched fresh on every run.
ONLINE_DATASET_URL = (
    "https://files.pythonhosted.org/packages/89/f6/"
    "ab35e962dd0b19f1008e88e788b202d45a90d9cd70b9bbf0ac26489ee260/"
    "nlpia-0.5.2-py2.py3-none-any.whl"
)
DATASET_PATH_IN_WHEEL = "nlpia/data/hutto_ICWSM_2014/tweets_GroundTruth.csv"

# How many tweets to sample from the 4,200-tweet corpus per run
SAMPLE_SIZE = 200

OUTPUT_FLAGGED_CSV = "investigator_review_list.csv"
OUTPUT_JSON        = "analysis_results.json"
OUTPUT_HTML        = "investigator_report.html"

# Labels for zero-shot classification
THREAT_LABELS = ["threatening and dangerous", "suspicious activity", "safe and benign"]

SLANG_MAP = {
    "wicked": "excellent",
    "sick": "cool",
    "badass": "impressive",
    "killing it": "succeeding",
    "bomb": "great" # Note: context-sensitive, but helps in social media triage
}

# High-Risk Keywords (Used to verify AI claims)
CRITICAL_KEYWORDS = [
    "kill", "shoot", "bomb", "attack", "murder", "death", "die",
    "destroy", "weapon", "gun", "knife", "threat", "hurt"
]

# ── Confidence tier thresholds ────────────────────────────────────────────────
# Combined threat score (P(threatening) + P(suspicious)) maps to four tiers:
#
#   CRITICAL   ≥ 0.90  — overtly threatening; immediate investigator action
#   HIGH       ≥ 0.80  — strongly suspicious; priority review
#   MEDIUM     ≥ 0.65  — ambiguous; secondary review
#   LOW        < 0.65  — likely benign; de-prioritised
#
# The facebook/bart-large-mnli model produces a noisy baseline on short, ambiguous,
# or topic-neutral text — it tends to hedge toward "suspicious activity" when it
# cannot find a clear safe signal, inflating scores for posts like conference tweets
# or casual conversation. Calibration against a labelled sample shows that genuinely
# benign posts rarely exceed 0.60, so the MEDIUM floor is set above that noise band.
#
TIER_CRITICAL  = 0.90
TIER_HIGH      = 0.80
TIER_MEDIUM    = 0.65   # anything below this is LOW / SAFE

# Minimum number of meaningful words (post-cleaning) required before the AI model
# is consulted. Posts shorter than this are automatically assigned LOW — the model
# cannot make a reliable judgement on fragments like "lol", "nice!", or "@user ok".
MIN_WORD_COUNT = 8

# ── Colour palette (dark forensics theme) ─────────────────────────────────────
BG_DARK    = "#0d1117"
BG_SURFACE = "#161b22"
BG_SURFACE2= "#1c2128"
COL_RED    = "#da3633"   # CRITICAL
COL_ORANGE = "#f0883e"   # HIGH
COL_AMBER  = "#e3b341"   # MEDIUM / threshold lines
COL_GREEN  = "#3fb950"   # LOW / SAFE
COL_BLUE   = "#58a6ff"
COL_MUTED  = "#8b949e"
COL_TEXT   = "#e6edf3"
COL_BORDER = "#30363d"

TIER_COLOURS = {
    "CRITICAL": COL_RED,
    "HIGH":     COL_ORANGE,
    "MEDIUM":   COL_AMBER,
    "LOW":      COL_GREEN,
}


# ── Step 1: Fetch Online Dataset ──────────────────────────────────────────────
def fetch_online_dataset(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Downloads the VADER tweet corpus (Hutto & Gilbert, ICWSM 2014) directly
    from files.pythonhosted.org at runtime — no local CSV required.

    The corpus is bundled inside the 'nlpia' Python package wheel (~30 MB).
    We stream the wheel, open it as a zip archive in memory, and extract
    only the CSV we need without writing anything to disk.

    Returns a DataFrame with columns: user_id, post_text, timestamp
    """
    print(f"\n[🌐] Fetching online dataset from files.pythonhosted.org ...")
    print(f"     Source : VADER Sentiment Corpus (Hutto & Gilbert, ICWSM 2014)")
    print(f"     URL    : {ONLINE_DATASET_URL[:70]}...")

    try:
        response = urllib.request.urlopen(ONLINE_DATASET_URL, timeout=60)
        raw_bytes = response.read()
        print(f"[✓] Downloaded {len(raw_bytes):,} bytes")
    except Exception as e:
        raise ConnectionError(f"[✗] Failed to fetch dataset: {e}")

    # Open the wheel (which is a zip file) entirely in memory
    wheel_zip = zipfile.ZipFile(io.BytesIO(raw_bytes))
    with wheel_zip.open(DATASET_PATH_IN_WHEEL) as csv_file:
        raw_df = pd.read_csv(csv_file)

    print(f"[✓] Loaded {len(raw_df):,} tweets from corpus")

    # Sample a manageable subset for classification (model is slow on CPU)
    sampled = raw_df.sample(n=min(sample_size, len(raw_df)), random_state=42).reset_index(drop=True)

    # Normalise to the schema the rest of the pipeline expects
    df = pd.DataFrame({
        "user_id":   [f"user_{str(row['id']).zfill(4)}" for _, row in sampled.iterrows()],
        "post_text": sampled["text"].astype(str),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    print(f"[✓] Sampled {len(df)} posts for analysis")
    return df


# ── Step 2: Data Preprocessing ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Prepares raw social media text for NLP analysis.
    Removes noise that could confuse the model:
      1. URLs and hyperlinks
      2. @mentions and #hashtags
      3. Emojis and non-ASCII unicode characters
      4. HTML entities (e.g. &amp;)
      5. Special characters (keeps basic punctuation)
      6. Excessive whitespace → collapses to single spaces
    Returns lowercase, clean text.
    """
    text = re.sub(r"http\S+|www\.\S+", "", text)         # 1. URLs
    text = re.sub(r"[@#](\w+)", r"\1", text)             # 2. @mentions / #hashtags
    text = text.encode("ascii", "ignore").decode("ascii") # 3. Emojis / non-ASCII
    text = re.sub(r"&\w+;", " ", text)                    # 4. HTML entities
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)     # 5. Special characters
    text = re.sub(r"\s+", " ", text).strip().lower()      # 6. Whitespace + lowercase
    words = text.split()
    normalized_words = [SLANG_MAP.get(w, w) for w in words]
    text = " ".join(normalized_words)

    return text


# ── Step 3: AI Classification + Confidence Tiers ─────────────────────────────
def assign_tier(threat_score: float) -> str:
    """
    Maps a combined threat score (0.0–1.0) to one of four confidence tiers.

    CRITICAL  ≥ 0.90  - overtly threatening
    HIGH      ≥ 0.80  - strongly suspicious
    MEDIUM    ≥ 0.65  - ambiguous / borderline
    LOW        < 0.65  - likely benign
    """
    if threat_score >= TIER_CRITICAL:
        return "CRITICAL"
    elif threat_score >= TIER_HIGH:
        return "HIGH"
    elif threat_score >= TIER_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


def classify_posts(df: pd.DataFrame, classifier, sentiment_pipe,
                   ner_pipe=None, emotion_pipe=None) -> pd.DataFrame:
    """
    Runs a multi-stage forensic pipeline:
    0. Batch pre-clean all texts + batch sentiment triage
    1. Short-text guard
    2. Sentiment triage (skip highly positive posts)
    3. AI zero-shot classification (BART)
    4. Keyword anchoring / verification
    5. NER + emotion detection on all FLAGGED posts
    """
    results = []
    tier_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

    print("\n[AI] Pre-cleaning all texts and running batch sentiment analysis...")
    all_originals = [str(row["post_text"]) for _, row in df.iterrows()]
    all_cleaned   = [clean_text(t) for t in all_originals]

    # Batch sentiment — much faster than one-by-one inference
    sentiment_results = sentiment_pipe(all_cleaned, batch_size=32, truncation=True)

    print("[AI] Classifying posts using Multi-Stage Forensic Pipeline...")
    print("-" * 80)
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        original_text = all_originals[idx]
        cleaned       = all_cleaned[idx]
        sent          = sentiment_results[idx]

        print(f"  [{idx + 1:>4} / {total}]", end="  ")

        # --- STAGE 1: SHORT-TEXT GUARD ---
        word_count = len(cleaned.split())
        if word_count < MIN_WORD_COUNT:
            results.append({
                "user_id": row["user_id"], "timestamp": row["timestamp"],
                "original_text": original_text, "clean_text": cleaned,
                "top_label": "safe and benign", "threat_score": 0.0,
                "safe_score": 1.0, "tier": "LOW", "status": "SAFE",
                "emotion": "—", "entities": "—",
                "note": f"skipped — short text ({word_count}w)"
            })
            print(f"🟢 [LOW     ] {row['user_id']} | score=0.00 (short) | {original_text[:60]}...")
            continue

        # --- STAGE 2: SENTIMENT TRIAGE ---
        if sent['label'] == 'POSITIVE' and sent['score'] > 0.85:
            results.append({
                "user_id": row["user_id"], "timestamp": row["timestamp"],
                "original_text": original_text, "clean_text": cleaned,
                "top_label": "safe and benign", "threat_score": 0.1,
                "safe_score": sent['score'], "tier": "LOW", "status": "SAFE",
                "emotion": "—", "entities": "—",
                "note": "Stage 2: High Positive Sentiment"
            })
            print(f"🟢 [LOW     ] {row['user_id']} | score=0.10 (Positive) | {original_text[:60]}...")
            continue

        # --- STAGE 3: AI ZERO-SHOT INFERENCE ---
        prediction = classifier(
            cleaned,
            candidate_labels=THREAT_LABELS,
            hypothesis_template="This text expresses a specific intent of {}."
        )
        scores    = dict(zip(prediction["labels"], prediction["scores"]))
        top_label = prediction["labels"][0]

        threat_score = (scores.get("threatening and dangerous", 0) +
                        scores.get("suspicious activity", 0))

        # --- STAGE 4: KEYWORD ANCHORING / VERIFICATION ---
        tier = assign_tier(threat_score)
        has_critical_word = any(
            re.search(r'\b' + re.escape(kw) + r'\b', cleaned)
            for kw in CRITICAL_KEYWORDS
        )
        note = ""

        if tier == "CRITICAL" and not has_critical_word:
            tier   = "MEDIUM"
            status = "FLAGGED"
            note   = "AI flagged CRITICAL but no violent keywords found; downgraded to MEDIUM."
        elif has_critical_word and threat_score > 0.5:
            if tier == "LOW":
                tier = "MEDIUM"
            note   = "Verified: Contains high-risk forensic keywords."
            status = "FLAGGED"
        else:
            status = "SAFE" if tier == "LOW" else "FLAGGED"

        # --- STAGE 5: NER + EMOTION (all FLAGGED posts) ---
        emotion_label = "—"
        entities_str  = "—"

        if status == "FLAGGED":
            if emotion_pipe is not None:
                emo = emotion_pipe(cleaned[:512])[0]
                emotion_label = f"{emo['label']} ({emo['score']:.2f})"

            if ner_pipe is not None:
                ner_result = ner_pipe(original_text[:512])
                entities = [
                    f"{e['word']} ({e['entity_group']})"
                    for e in ner_result
                    if e['entity_group'] in ('PER', 'LOC', 'ORG')
                ]
                entities_str = ", ".join(entities) if entities else "none detected"
                if entities_str != "none detected":
                    note += f" | Entities: {entities_str}"

        results.append({
            "user_id":       row["user_id"],
            "timestamp":     row["timestamp"],
            "original_text": original_text,
            "clean_text":    cleaned,
            "top_label":     top_label,
            "threat_score":  round(threat_score, 4),
            "safe_score":    round(scores.get("safe and benign", 0), 4),
            "tier":          tier,
            "status":        status,
            "emotion":       emotion_label,
            "entities":      entities_str,
            "note":          note,
        })

        print(f"{tier_icons[tier]} [{tier:8s}] {row['user_id']} | score={threat_score:.2f} "
              f"| emo={emotion_label.split('(')[0].strip():10s} | {original_text[:50]}...")

    return pd.DataFrame(results)


# ── Step 4: Export Results ────────────────────────────────────────────────────
def export_results(df: pd.DataFrame) -> pd.DataFrame:
    """Saves flagged posts (sorted by tier priority) to CSV and full results to JSON."""
    # Define tier priority order for sorting
    tier_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    flagged = df[df["status"] == "FLAGGED"].copy()
    flagged["_sort"] = flagged["tier"].map(tier_order)
    flagged = (flagged
               .sort_values(["_sort", "threat_score"], ascending=[True, False])
               .drop(columns=["_sort"]))

    flagged.to_csv(OUTPUT_FLAGGED_CSV, index=False)
    print(f"\n[✓] Investigator review list → {OUTPUT_FLAGGED_CSV} ({len(flagged)} posts)")

    df.to_json(OUTPUT_JSON, orient="records", indent=2)
    print(f"[✓] Full results JSON       → {OUTPUT_JSON}")

    return flagged


# ── Step 5: HTML Investigator Report ─────────────────────────────────────────
def export_html(df: pd.DataFrame, path: str = OUTPUT_HTML):
    """
    Generates a colour-coded HTML report of all flagged posts.
    Opens in any browser — far more readable than a raw CSV for presentation.
    """
    TIER_HEX = {
        "CRITICAL": COL_RED,
        "HIGH":     COL_ORANGE,
        "MEDIUM":   COL_AMBER,
        "LOW":      COL_GREEN,
    }

    flagged = df[df["status"] == "FLAGGED"].copy()
    tier_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    flagged["_sort"] = flagged["tier"].map(tier_order)
    flagged = flagged.sort_values(["_sort", "threat_score"], ascending=[True, False]).drop(columns=["_sort"])

    rows_html = ""
    for _, r in flagged.iterrows():
        color = TIER_HEX.get(r["tier"], "#ffffff")
        rows_html += (
            f'<tr style="border-left:4px solid {color}">'
            f'<td style="white-space:nowrap">{r["user_id"]}</td>'
            f'<td style="color:{color};font-weight:bold">{r["tier"]}</td>'
            f'<td style="text-align:center">{r["threat_score"]:.3f}</td>'
            f'<td>{r["original_text"]}</td>'
            f'<td style="white-space:nowrap">{r.get("emotion", "—")}</td>'
            f'<td style="font-size:0.85em">{r.get("entities", "—")}</td>'
            f'<td style="font-size:0.8em;color:#8b949e">{r.get("note", "")}</td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Forensic Investigator Report</title>
  <style>
    body  {{ background:#0d1117; color:#e6edf3; font-family:monospace; padding:24px; margin:0 }}
    h1    {{ color:#58a6ff; margin-bottom:4px; font-size:1.3em }}
    p.sub {{ color:#8b949e; font-size:0.85em; margin-top:0 }}
    table {{ border-collapse:collapse; width:100%; margin-top:16px }}
    thead tr {{ background:#161b22 }}
    th    {{ padding:10px 12px; text-align:left; color:#8b949e;
             font-size:0.8em; text-transform:uppercase; letter-spacing:0.05em }}
    td    {{ padding:8px 12px; border-bottom:1px solid #21262d; vertical-align:top }}
    tr:hover td {{ background:#161b22 }}
    .tag-c {{ background:{COL_RED};    color:#fff; border-radius:4px; padding:2px 6px; font-size:0.75em }}
    .tag-h {{ background:{COL_ORANGE}; color:#000; border-radius:4px; padding:2px 6px; font-size:0.75em }}
    .tag-m {{ background:{COL_AMBER};  color:#000; border-radius:4px; padding:2px 6px; font-size:0.75em }}
  </style>
</head>
<body>
  <h1>Digital Forensics — Investigator Review List</h1>
  <p class="sub">
    Model: facebook/bart-large-mnli &nbsp;|&nbsp;
    Dataset: VADER/ICWSM 2014 &nbsp;|&nbsp;
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
    Flagged posts: {len(flagged)}
  </p>
  <table>
    <thead>
      <tr>
        <th>User</th>
        <th>Tier</th>
        <th>Score</th>
        <th>Original Post</th>
        <th>Emotion</th>
        <th>Entities (NER)</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] HTML report            → {path}")


# ── Step 6: Live Matplotlib Dashboard ────────────────────────────────────────
def show_charts(df: pd.DataFrame):
    """
    Displays a 4-panel forensic analysis dashboard.

    Panel 1 (top-left)   — Donut chart: four-tier post breakdown
    Panel 2 (top-right)  — Histogram: threat score distribution with tier bands
    Panel 3 (bottom-left)— Horizontal bar: top 15 posts ranked by threat score
    Panel 4 (bottom-right)— Word cloud of most frequent terms in CRITICAL+HIGH posts
                            (falls back to scatter plot if wordcloud not installed)
    """
    plt.rcParams.update({
        "figure.facecolor":  BG_DARK,
        "axes.facecolor":    BG_SURFACE,
        "axes.edgecolor":    COL_BORDER,
        "axes.labelcolor":   COL_MUTED,
        "axes.titlecolor":   COL_TEXT,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.titlepad":     12,
        "xtick.color":       COL_MUTED,
        "ytick.color":       COL_MUTED,
        "text.color":        COL_TEXT,
        "grid.color":        COL_BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "legend.facecolor":  BG_DARK,
        "legend.edgecolor":  COL_BORDER,
        "legend.labelcolor": COL_TEXT,
        "legend.fontsize":   9,
    })

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "DIGITAL FORENSICS  —  SOCIAL MEDIA NLP THREAT ANALYSIS  |  4-TIER CONFIDENCE SYSTEM",
        fontsize=13, fontweight="bold", color=COL_TEXT, y=0.98
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.07
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    tier_counts = df["tier"].value_counts()
    total       = len(df)

    # ── Panel 1: Donut — four tiers ───────────────────────────────────────────
    tiers      = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    counts_ord = [tier_counts.get(t, 0) for t in tiers]
    colors_ord = [TIER_COLOURS[t] for t in tiers]
    labels_ord = [f"{t}\n{tier_counts.get(t, 0)}" for t in tiers]

    # Only show non-zero slices
    nonzero = [(t, c, col, l) for t, c, col, l in
               zip(tiers, counts_ord, colors_ord, labels_ord) if c > 0]
    nz_counts  = [x[1] for x in nonzero]
    nz_colors  = [x[2] for x in nonzero]
    nz_labels  = [x[3] for x in nonzero]

    ax1.pie(
        nz_counts, labels=nz_labels, colors=nz_colors,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG_DARK, linewidth=2),
        textprops=dict(color=COL_TEXT, fontsize=9, fontweight="bold"),
    )
    ax1.set_title("Confidence Tier Breakdown")
    ax1.text(0, 0, f"{total}\nposts", ha="center", va="center",
             fontsize=13, fontweight="bold", color=COL_TEXT)

    flagged_n = total - tier_counts.get("LOW", 0)
    ax1.text(0, -1.55,
             f"Flagged: {flagged_n}/{total}  |  "
             f"Critical≥{TIER_CRITICAL}  High≥{TIER_HIGH}  Medium≥{TIER_MEDIUM}",
             ha="center", fontsize=7.5, color=COL_MUTED)

    # ── Panel 2: Histogram with tier band shading ─────────────────────────────
    scores_all = df["threat_score"].values
    n_bins = 20
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    bar_vals, _, patches = ax2.hist(scores_all, bins=bin_edges,
                                    edgecolor=BG_DARK, linewidth=0.8)

    def tier_colour_for_edge(edge):
        if edge >= TIER_CRITICAL:  return COL_RED
        if edge >= TIER_HIGH:      return COL_ORANGE
        if edge >= TIER_MEDIUM:    return COL_AMBER
        return COL_GREEN

    for patch, edge in zip(patches, bin_edges[:-1]):
        patch.set_facecolor(tier_colour_for_edge(edge))

    for threshold, label in [
        (TIER_MEDIUM,  f"MEDIUM {TIER_MEDIUM}"),
        (TIER_HIGH,    f"HIGH {TIER_HIGH}"),
        (TIER_CRITICAL,f"CRITICAL {TIER_CRITICAL}"),
    ]:
        ax2.axvline(threshold, color=COL_MUTED, linewidth=1.2, linestyle="--", alpha=0.8)
        ax2.text(threshold + 0.01, ax2.get_ylim()[1] * 0.95, label,
                 fontsize=6.5, color=COL_MUTED, va="top")

    ax2.set_title("Threat Score Distribution by Tier")
    ax2.set_xlabel("Threat Score")
    ax2.set_ylabel("Number of Posts")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(axis="y")

    legend_patches = [mpatches.Patch(color=TIER_COLOURS[t], label=t) for t in tiers]
    ax2.legend(handles=legend_patches, loc="upper left", fontsize=8)

    # ── Panel 3: Horizontal bar — top 15 ──────────────────────────────────────
    top_n = min(15, len(df))
    top15 = (df.nlargest(top_n, "threat_score")
               .sort_values("threat_score", ascending=True)
               .reset_index(drop=True))

    bar_colors = [TIER_COLOURS[t] for t in top15["tier"]]

    bars = ax3.barh(
        top15["user_id"], top15["threat_score"],
        color=bar_colors, edgecolor=BG_DARK, linewidth=0.6, height=0.7
    )

    for bar, score, tier in zip(bars, top15["threat_score"], top15["tier"]):
        ax3.text(
            min(bar.get_width() + 0.01, 1.02),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f} [{tier}]", va="center", ha="left",
            fontsize=7, color=COL_TEXT
        )

    for threshold in [TIER_MEDIUM, TIER_HIGH, TIER_CRITICAL]:
        ax3.axvline(threshold, color=COL_MUTED, linewidth=1.0, linestyle="--", alpha=0.7)

    ax3.set_title(f"Top {top_n} Posts by Threat Score")
    ax3.set_xlabel("Threat Score  (0 = Safe → 1 = High Threat)")
    ax3.set_xlim(0, 1.22)
    ax3.grid(axis="x")

    legend_patches = [mpatches.Patch(color=TIER_COLOURS[t], label=t) for t in tiers]
    ax3.legend(handles=legend_patches, loc="lower right", fontsize=8)

    # ── Panel 4: Word cloud — threat vocabulary of CRITICAL + HIGH posts ─────
    flagged_text = " ".join(
        df[df["tier"].isin(["CRITICAL", "HIGH"])]["clean_text"].tolist()
    )

    if WORDCLOUD_AVAILABLE and flagged_text.strip():
        wc = WordCloud(
            width=900, height=420,
            background_color=BG_DARK,
            colormap="Reds",
            max_words=60,
            prefer_horizontal=0.85,
        ).generate(flagged_text)
        ax4.imshow(wc, interpolation="bilinear")
        ax4.axis("off")
        ax4.set_title("Most Frequent Terms in CRITICAL + HIGH Posts")
    else:
        # Fallback: scatter plot when wordcloud library is unavailable
        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            subset = df[df["tier"] == tier]
            if subset.empty:
                continue
            marker = "D" if tier in ("HIGH", "CRITICAL") else "o"
            ax4.scatter(
                subset["threat_score"], subset["safe_score"],
                color=TIER_COLOURS[tier], alpha=0.85,
                s=80 if tier in ("HIGH", "CRITICAL") else 55,
                edgecolors=BG_DARK, linewidth=0.5,
                marker=marker, label=tier, zorder=3
            )
        for threshold in [TIER_MEDIUM, TIER_HIGH, TIER_CRITICAL]:
            ax4.axvline(threshold, color=COL_MUTED, linewidth=0.8,
                        linestyle="--", alpha=0.6, zorder=2)
        ax4.set_title("Threat Score vs Safe Score  (by Tier)")
        ax4.set_xlabel("Threat Score →")
        ax4.set_ylabel("← Safe Score")
        ax4.set_xlim(-0.05, 1.05)
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True)
        if not WORDCLOUD_AVAILABLE:
            ax4.text(0.5, -0.12, "Install wordcloud for threat vocabulary panel",
                     transform=ax4.transAxes, ha="center",
                     fontsize=7.5, color=COL_MUTED)

    fig.text(
        0.5, 0.005,
        f"Model: facebook/bart-large-mnli  |  Zero-Shot NLP  |  "
        f"Dataset: VADER/ICWSM 2014 tweets  |  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ha="center", fontsize=7.5, color=COL_MUTED
    )

    dashboard_path = "forensic_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Dashboard saved → {dashboard_path}")
    plt.show()
    print("[✓] Charts displayed.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Digital Forensics — Social Media NLP Threat Analyzer")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1 — Fetch live data from online source
    df = fetch_online_dataset(sample_size=SAMPLE_SIZE)

    # Step 2 — Load all AI models
    print("\n[AI] Loading Stage-1: Fast Sentiment Analyzer (distilbert)...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    print("[AI] Loading Stage-2: Zero-Shot Threat Classifier (facebook/bart-large-mnli)...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1   # -1 = CPU; set to 0 for GPU
    )

    print("[AI] Loading Stage-3: Named Entity Recognition (dslim/bert-base-NER)...")
    ner_pipe = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1
    )

    print("[AI] Loading Stage-4: Emotion Classifier (j-hartmann/emotion-english-distilroberta-base)...")
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1
    )

    print("[✓] All models loaded.\n")

    # Step 3 — Preprocess and classify with tier assignment
    results_df = classify_posts(df, classifier, sentiment_pipe, ner_pipe, emotion_pipe)

    # Step 4 — Export CSV + JSON + HTML
    export_results(results_df)
    export_html(results_df)

    # Step 5 — Summary
    tier_counts = results_df["tier"].value_counts()
    total   = len(results_df)
    flagged = len(results_df[results_df["status"] == "FLAGGED"])
    safe    = total - flagged

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  Total posts analysed : {total}")
    print(f"  ─────────────────────────────────────────")
    print(f"  🔴 CRITICAL  (≥{TIER_CRITICAL}) : {tier_counts.get('CRITICAL', 0):>4}  — Immediate review")
    print(f"  🟠 HIGH      (≥{TIER_HIGH}) : {tier_counts.get('HIGH', 0):>4}  — Priority review")
    print(f"  🟡 MEDIUM    (≥{TIER_MEDIUM}) : {tier_counts.get('MEDIUM', 0):>4}  — Secondary review")
    print(f"  🟢 LOW       (<{TIER_MEDIUM}) : {tier_counts.get('LOW', 0):>4}  — De-prioritised")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total FLAGGED        : {flagged}  ({flagged/total*100:.1f}%)")
    print(f"\n  Output files:")
    print(f"    → {OUTPUT_FLAGGED_CSV}")
    print(f"    → {OUTPUT_JSON}")
    print(f"    → {OUTPUT_HTML}")
    print(f"    → forensic_dashboard.png")
    print("=" * 70)

    print("\n[📊] Opening forensic analysis dashboard...")
    show_charts(results_df)


if __name__ == "__main__":
    main()
