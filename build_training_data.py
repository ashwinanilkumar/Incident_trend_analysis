"""
build_training_data.py
======================
Reads Last12months_incident_dump.csv, classifies each incident using
ONLY the deterministic rules pipeline (no ML/similarity guessing),
and saves the high-confidence results to historical/ as a new Excel file.

This expands the training set for the similarity matcher and ML classifier
without introducing errors from probabilistic models.

Usage:
    python build_training_data.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd
from app import (apply_rules, load_rules, load_synonyms,
                 expand_text_with_synonyms, combine_text,
                 normalize_module, VALID_ISSUE_TYPES, VALID_MODULES)

CSV_PATH = "Last12months_incident_dump.csv"
OUT_PATH = os.path.join("historical", "Last12months_classified.xlsx")

# Only keep rule-based classifications at or above this priority threshold
# (high-priority rules are the most reliable)
MIN_PRIORITY = 75
# Cap BigPanda rows so they don't dominate the training set
BIGPANDA_CAP = 200


def main():
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    print(f"  Total rows: {len(df)}")

    rules = load_rules()
    synonyms = load_synonyms()
    print(f"  Rules loaded: {len(rules)}")

    results = []
    skipped = 0
    bigpanda_count = 0

    for i, row in df.iterrows():
        text = combine_text(row)
        expanded = expand_text_with_synonyms(text, synonyms)

        module_raw, type_issue, kw, priority = apply_rules(expanded, rules)

        if module_raw is None or priority < MIN_PRIORITY:
            skipped += 1
            continue

        module = normalize_module(module_raw)
        if module not in VALID_MODULES:
            skipped += 1
            continue

        # Cap BigPanda to avoid imbalance
        if module == "BigPanda":
            if bigpanda_count >= BIGPANDA_CAP:
                skipped += 1
                continue
            bigpanda_count += 1

        # Only keep types that are in the valid taxonomy
        if type_issue not in VALID_ISSUE_TYPES:
            skipped += 1
            continue

        results.append({
            "Number": row.get("Number", ""),
            "Short Description": row.get("Short Description", ""),
            "Description": row.get("Description", ""),
            "Module": module,
            "Type Of Issue": type_issue,
            "Classification Method": f"rule ({kw})",
            "Rule Priority": priority,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(df)} rows, "
                  f"classified: {len(results)}, skipped: {skipped}")

    print(f"\nClassification complete:")
    print(f"  Classified (rule-based, priority >= {MIN_PRIORITY}): {len(results)}")
    print(f"  Skipped (low priority / unknown / BigPanda cap): {skipped}")

    if not results:
        print("No results to save.")
        return

    out_df = pd.DataFrame(results)

    # Show per-module breakdown
    print("\nPer-module breakdown:")
    for mod, cnt in out_df["Module"].value_counts().items():
        print(f"  {mod:25s}: {cnt}")

    # Save to historical folder
    os.makedirs("historical", exist_ok=True)
    out_df.to_excel(OUT_PATH, index=False, sheet_name="Dump")
    print(f"\nSaved {len(out_df)} rows to {OUT_PATH}")
    print("Restart the Streamlit app to pick up the new training data.")


if __name__ == "__main__":
    main()
