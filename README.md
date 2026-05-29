# RACPad Incident Trend Analysis — Classification Tool

A Streamlit-based web application that automatically classifies RACPad support incidents into **modules** and **issue types** using a multi-stage hybrid pipeline, with built-in correction feedback, trend comparison, and export capabilities.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Classification Pipeline](#classification-pipeline)
- [Configuration Files](#configuration-files)
- [Historical Training Data](#historical-training-data)
- [Exporting Results](#exporting-results)

---

## Overview

This tool processes Excel exports from ServiceNow / RACPad incident queues and automatically classifies each incident with:

- **Module** — the application area (e.g. Payment, Inventory, Agreement, BigPanda)
- **Type Of Issue** — the specific problem category (e.g. Payment Error, Inventory Transfer Issue)
- **Confidence** — High / Medium / Low - Needs Review
- **Classification Method** — which pipeline stage made the decision

Users can review, correct, and confirm classifications inline. Every correction is learned and applied to future runs via an auto-promotion engine.

---

## Features

| Feature | Description |
|---|---|
| **Multi-stage pipeline** | Rules → Historical Similarity → ML Prediction → AI Review → Semantic Review |
| **BigPanda classification** | Auto-detects BigPanda incidents and extracts service name as issue type |
| **Inline corrections** | Edit Module/Type Of Issue directly in the results table — type custom values or use existing ones |
| **Custom taxonomy** | New modules/issue types entered by users are persisted to `rules_config.json` automatically |
| **Learned corrections** | Corrections are saved and applied to future classifications (fuzzy match + auto-promotion to rules) |
| **AI Review** | Calibrated SVM second-opinion reviewer promotes low/medium confidence incidents |
| **Semantic Review** | Neural language model (sentence-transformers) validates all incidents for plausibility |
| **Trend Comparison** | Compare 2–3 time-period dumps side by side with interactive charts |
| **BigPanda Dashboard** | Dedicated section showing BigPanda incidents broken down by service/application |
| **Excel Export** | Download classified results as a formatted .xlsx with all classification metadata |
| **Historical Data Management** | Upload and manage training data that improves the ML model |
| **Auto-promoted Rules** | Patterns seen ≥3 times in corrections are automatically promoted to permanent rules |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` / `openpyxl` | Excel reading, data manipulation |
| `scikit-learn` | TF-IDF vectorizer, Logistic Regression, LinearSVC, cosine similarity |
| `rapidfuzz` | Fuzzy keyword matching for rule engine |
| `sentence-transformers` | Neural semantic review (optional — degrades gracefully if not available) |
| `plotly` | Interactive charts (trend comparison, confidence distribution, module breakdown) |
| `numpy` | Numerical operations |

---

## Project Structure

```
Incident Trend Analysis/
├── app.py                  # Main Streamlit application (all logic)
├── rules_config.json       # Hand-crafted classification rules + valid taxonomy lists
├── corrections.json        # Saved user corrections + auto-promoted rules (auto-generated)
├── synonyms.json           # Synonym/keyword expansion dictionary
├── requirements.txt        # Python dependencies
├── setup.bat               # One-click setup script (Windows) — installs Python + venv
├── run.bat                 # Launch script (Windows)
├── prebuild_models.py      # Optional: pre-build ML models outside the app
├── build_training_data.py  # Utility: build training data from historical files
├── historical/             # Historical classified Excel files (training data)
│   ├── Jan_Ticket_Analysis.xlsx
│   ├── February Ticket Analysis.xlsx
│   └── ...
└── scripts/
    └── make_workspace_zip.py   # Utility: package app for distribution
```

---

## Setup & Installation

### Windows (Recommended — One Click)

1. **Clone or download** this repository to your local machine.
2. Double-click **`setup.bat`**.
   - Automatically detects or downloads Python 3.11.
   - Creates a virtual environment (`venv/`).
   - Installs all dependencies from `requirements.txt`.
3. When setup completes, proceed to [Running the App](#running-the-app).

### Manual Setup (any OS)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
#    Windows:
venv\Scripts\activate
#    macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-build ML models to avoid first-run delay
python prebuild_models.py
```

> **Note:** The `sentence-transformers` library downloads a ~90 MB language model on first use. Ensure internet access on the first run, or pre-download by running the app once.

---

## Running the App

### Windows

Double-click **`run.bat`** — the app opens at [http://localhost:8501](http://localhost:8501).

### Any OS

```bash
# Activate the virtual environment first, then:
streamlit run app.py
```

---

## Usage Guide

### 1. Classification Tool Tab

1. **Upload** an Excel file (`.xlsx`) containing at minimum:
   - `Short Description` column
   - `Description` column
   - Optionally: `Number` (incident number) for exact-match corrections
2. Click **Analyze & Classify Incidents**.
3. Review the results:
   - **Items Still Needing Review** — Low/Medium confidence incidents that need human verification.
   - **Classification Analytics** — Confidence distribution, method breakdown, module distribution.
   - **BigPanda Incidents** — Dedicated section for BigPanda monitoring alerts.
   - **Classified Data (All)** — Full table, fully editable.
4. **Correct misclassifications** by clicking any Module or Type Of Issue cell and typing a new value (you can enter custom values not in the predefined list).
5. Click **Save Corrections** — the system learns from your feedback.
6. **Download** the classified Excel file.

### 2. Trend Comparison Tab

- Upload 2 or 3 time-period Excel files (pre-classified or raw).
- View side-by-side comparison charts showing volume changes by module and issue type.

### 3. Manage Training Data Tab

- Upload new classified Excel files to the historical training set.
- Remove outdated training files.
- Rebuild ML models after adding new data.

---

## Classification Pipeline

Each incident passes through these stages in order. The first stage that produces a confident result wins; later stages only run for unresolved incidents.

```
Stage -1  BigPanda Fast-Path
          └─ If "bigpanda" is in Short Description → Module=BigPanda,
             Type=[service_name] - Backend 500 Errors / Service Outage

Stage  0  Learned Corrections
          └─ Exact incident number match OR fuzzy text match to past corrections

Stage  1  Rule Engine (Exact)
          └─ Keyword matching against hand-crafted rules in rules_config.json
             Rules are sorted by priority (highest first); longer keywords win

Stage  1b Rule Engine (Fuzzy)
          └─ RapidFuzz token matching when exact match fails

Stage  2  Historical Similarity
          └─ TF-IDF cosine similarity against 9,800+ historical classified incidents

Stage  3  ML Prediction
          └─ Logistic Regression trained on historical data

Stage  4  AI Review (optional)
          └─ Calibrated LinearSVC second-opinion on Low/Medium confidence results
             Can promote incidents to High confidence

Stage  5  Semantic Review (optional)
          └─ sentence-transformers neural model validates all results
             Can override module assignment or flag for human review
```

---

## Configuration Files

### `rules_config.json`

Contains three sections:

```json
{
  "modules": ["Dashboard", "Customer", "Payment", ...],
  "issue_types": ["Payment Error", "Inventory Transfer Issue", ...],
  "rules": [
    {
      "keywords": ["bigpanda incident", "bigpanda"],
      "module": "BigPanda",
      "type_of_issue": "BigPanda",
      "priority": 200,
      "description": "BigPanda monitoring alerts"
    }
  ]
}
```

- **`modules`** / **`issue_types`**: Valid taxonomy values. New values entered by users via the UI are appended here automatically.
- **`rules`**: Each rule has `keywords`, `module`, `type_of_issue`, and `priority`. Higher priority = checked first.

### `corrections.json`

Auto-generated. Stores all user corrections and auto-promoted rules. **Do not edit manually.** This file is the app's memory — deleting it resets all learned corrections.

### `synonyms.json`

Maps synonyms and shorthand phrases to canonical terms used during text preprocessing (e.g. `"RTO" → "rent-to-own"`).

---

## Historical Training Data

Place classified Excel files (with `Module` and `Type Of Issue` columns populated) in the `historical/` folder. The ML model trains on all files in this folder. The more diverse and accurate the historical data, the better the ML predictions.

**Minimum required columns in training files:**
- `Short Description`
- `Description`
- `Module`
- `Type Of Issue`

---

## Exporting Results

After classification, click **Download Classified Excel**. The exported file includes:

| Column | Description |
|---|---|
| All original columns | Unchanged from your upload |
| `Module` | Classified module |
| `Type Of Issue` | Classified issue type |
| `Confidence` | High / Medium / Low - Needs Review |
| `Classification Method` | Which pipeline stage classified this row |
| `AI Review` | AI reviewer result (if enabled) |
| `AI Confidence` | AI reviewer confidence score |
| `AI Reason` | AI reviewer reasoning |
| `Semantic Review` | Semantic model result (if enabled) |
| `Semantic Confidence` | Semantic model confidence |
| `Semantic Reason` | Semantic model reasoning |

---

## Team Notes

- **First run** trains the ML models (~3–5 min). Subsequent runs load from cache (seconds).
- The `corrections.json` file is the most valuable asset — back it up regularly.
- To force a model rebuild (after adding new historical data), delete the `.cache/` folder.
- The sidebar lets you tune similarity and confidence thresholds, and toggle AI/Semantic Review on/off.
