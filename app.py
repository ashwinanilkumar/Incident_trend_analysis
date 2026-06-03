import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import hashlib
import pickle
import zipfile
import warnings
from collections import Counter
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore")


# ==================== ROBUST EXCEL READER ====================
# Some Excel files use old OOXML namespaces (purl.oclc.org) that openpyxl
# can't recognize. This function patches the namespaces before reading.
_NS_MAP = [
    (b"http://purl.oclc.org/ooxml/spreadsheetml/main",
     b"http://schemas.openxmlformats.org/spreadsheetml/2006/main"),
    (b"http://purl.oclc.org/ooxml/officeDocument/relationships",
     b"http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    (b"http://purl.oclc.org/ooxml/drawingml/main",
     b"http://schemas.openxmlformats.org/drawingml/2006/main"),
]


def _patch_xlsx_namespaces(file_bytes: bytes) -> bytes:
    """Rewrite old purl.oclc.org namespaces to standard openxmlformats.org ones."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(file_bytes), "r") as zin, \
         zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".xml") or item.filename.endswith(".rels"):
                for old_ns, new_ns in _NS_MAP:
                    data = data.replace(old_ns, new_ns)
            zout.writestr(item, data)
    return out.getvalue()


def read_excel_robust(file_bytes: bytes, sheet_name: str = None) -> tuple:
    """
    Read an Excel file from bytes, patching namespace issues if needed.
    Returns (DataFrame, sheet_name, all_sheet_names).
    If sheet_name is None, auto-selects 'Dump' or the first sheet.
    """
    def _try_read(data: bytes):
        xl = pd.ExcelFile(BytesIO(data), engine="openpyxl")
        return xl.sheet_names

    # First attempt: standard read
    sheet_names = []
    patched_bytes = file_bytes
    try:
        sheet_names = _try_read(file_bytes)
    except Exception:
        pass

    # Second attempt: patch namespaces
    if not sheet_names:
        try:
            patched_bytes = _patch_xlsx_namespaces(file_bytes)
            sheet_names = _try_read(patched_bytes)
        except Exception:
            pass

    if not sheet_names:
        raise ValueError(
            "Could not read the Excel file. "
            "Ensure it is a valid .xlsx file with at least one sheet."
        )

    # Determine which sheet to read
    if sheet_name is None:
        sheet_name = "Dump" if "Dump" in sheet_names else sheet_names[0]

    df = pd.read_excel(BytesIO(patched_bytes), sheet_name=sheet_name, engine="openpyxl")
    return df, sheet_name, sheet_names

# ==================== CONFIG ====================
RULES_FILE = "rules_config.json"
SYNONYMS_FILE = "synonyms.json"
CORRECTIONS_FILE = "corrections.json"
CACHE_DIR = ".cache"
HISTORICAL_DIR = "historical"
SIMILARITY_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD = 0.40
FUZZY_THRESHOLD = 82  # Minimum fuzzy match score (0-100)
RANDOM_STATE = 42

# Valid modules and issue types (from RACPad taxonomy)
VALID_MODULES = [
    "Dashboard", "Customer", "Agreement", "Payment", "Inventory",
    "Account Management", "Store", "Operations", "Reporting",
    "Menu / Container", "Customer Service", "Data Services", "BigPanda"
]

# New standard types (from reference Excel)
VALID_ISSUE_TYPES = [
    # Inventory
    "Switch Out Issue", "Unable to receive the Purchase order",
    "Unable to reverse PO", "Cancel the Manual PO", "Cancel the RMS PO",
    "PO related issue", "Inventory Issue", "Inventory Status Issue",
    "Inventory Transfer Issue", "Inventory Mismatch", "DQ-Inventory", "Chargeoff",
    # Agreement
    "Confirm delivery Issue",
    "Agreement Issue", "Agreement Status", "Agreement Creation Issue",
    "Agreement Void Issue", "Agreement Transfer Issue", "Agreement delivery Issue",
    "DQ-Agreement", "RAC exchange",
    # Payment
    "Payment Error", "Payment Query", "Payment issue", "Payment reversal",
    "Payment mode Issue", "AutoPay", "Chargeback", "Receipt Mismatch",
    "Receipt Reversal Issue", "Tax Issue", "Unable to Save card details",
    # Customer
    "Customer Issue", "Customer Login Issue (EXT)", "Customer Mismatch",
    "Customer level change request", "Merge Customer", "Duplicate customers record",
    "search Issue",
    # Store / Operations
    "Store Access", "Menu Permission Access", "Drop reconciliation",
    "PIN reset", "Daily task", "Training Issue",
    # Reporting
    "Report discrepancy", "Daily Report Issue", "Report Request",
    "Account Report Request", "Report access", "EOM",
    # Pricing / Promo
    "Pricing Issue", "Pricing request", "Promo issue", "Promo request",
    # Operations / Misc
    "2 way sync", "Monitoring", "Duplicate ticket", "Intermittent Issue",
    "One Time Issue", "Query", "Request",
    # Special
    "BigPanda",
    "The EPO amount is incorrectly calculated as 0 after the SAC period ends.",
    "Customer information issue",
    # Legacy types (kept for backward compatibility with historical training data)
    "Business Logic Error", "Data Discrepancy", "Payment Processing Failure",
    "Inventory Management Error", "API / Service Error", "Access / Permission Issue",
    "Authentication / SSO Issue", "Data Sync / Integration Failure",
    "Configuration Issue", "Document / Receipt Error", "UI/Display Error",
    "Performance / Timeout", "Translation / Localization", "EOD / Batch Job Failure",
    "CCCB / Chargeback",
]

# Module normalization mapping (maps old/variant names to canonical)
MODULE_NORMALIZATION = {
    "payment": "Payment",
    "customer": "Customer",
    "agreement": "Agreement",
    "agreement ": "Agreement",
    "store": "Store",
    "inventory": "Inventory",
    "operations": "Operations",
    "report": "Reporting",
    "reports": "Reporting",
    "reporting": "Reporting",
    "dashboard": "Dashboard",
    "admin": "Store",
    "app config": "Operations",
    "access": "Store",
    "database": "Data Services",
    "cash management": "Store",
    "pricing": "Operations",
    "bigpanda": "BigPanda",
    "big panda": "BigPanda",
    "other": "Operations",
    "account management": "Account Management",
}


# ==================== PREPROCESSING ====================
def preprocess_text(text):
    """Lowercase, remove symbols, clean whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text(row):
    """Combine Short Description + Description into single text."""
    short_desc = preprocess_text(row.get("Short Description", ""))
    desc = preprocess_text(row.get("Description", ""))
    return f"{short_desc} {desc}".strip()


# ==================== SYNONYM & DICTIONARY EXPANSION ====================
@st.cache_data(show_spinner=False)
def load_synonyms(synonyms_file=SYNONYMS_FILE):
    """Load synonym dictionary and custom expansions (cached — file rarely changes)."""
    if os.path.exists(synonyms_file):
        with open(synonyms_file, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {"synonyms": {}, "custom_expansions": {}}


def expand_text_with_synonyms(text, synonym_config):
    """
    Expand text by appending synonym terms found in the text.
    Also expands abbreviations from custom_expansions.
    """
    expanded = text
    expansions = synonym_config.get("custom_expansions", {})
    synonyms = synonym_config.get("synonyms", {})

    # Expand abbreviations (word boundary matching)
    for abbr, full_form in expansions.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        if re.search(pattern, expanded):
            expanded = expanded + " " + full_form

    # Append synonym matches to boost matching
    for base_word, syn_list in synonyms.items():
        for syn in syn_list:
            if syn.lower() in expanded and base_word.lower() not in expanded:
                expanded = expanded + " " + base_word
                break

    return expanded


# ==================== FUZZY MATCHING ====================
def fuzzy_match_keyword(text, keywords, threshold=FUZZY_THRESHOLD):
    """
    Try fuzzy matching if exact keyword match fails.
    Uses token_set_ratio for partial matching on multi-word keywords.
    Returns (matched_keyword, score) or (None, 0).
    """
    # Sort keywords by length descending (prefer specific matches)
    sorted_keywords = sorted(keywords, key=len, reverse=True)

    for keyword in sorted_keywords:
        # For multi-word keywords, use token_set_ratio
        if len(keyword.split()) > 1:
            score = fuzz.token_set_ratio(keyword.lower(), text.lower())
        else:
            # For single words, check if any word in text is close
            words_in_text = text.lower().split()
            score = 0
            for word in words_in_text:
                s = fuzz.ratio(keyword.lower(), word)
                if s > score:
                    score = s
        if score >= threshold:
            return keyword, score

    return None, 0


# ==================== CORRECTIONS / LEARNING ====================
def load_corrections(corrections_file=CORRECTIONS_FILE):
    """Load correction history, deduplicating exact incident-number entries (latest wins)."""
    if os.path.exists(corrections_file):
        with open(corrections_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        all_corrections = data.get("corrections", [])
        # For the same incident_number submitted multiple times, keep only the latest.
        # Non-numbered corrections (submitted via the manual form without an INC#) are kept as-is.
        numbered = {}
        non_numbered = []
        for corr in all_corrections:
            num = str(corr.get("incident_number", "")).strip().upper()
            if num:
                numbered[num] = corr  # later entries overwrite earlier ones
            else:
                non_numbered.append(corr)
        return list(numbered.values()) + non_numbered
    return []


def save_correction(incident_number, original_module, original_type, corrected_module,
                    corrected_type, short_description, description="",
                    corrections_file=CORRECTIONS_FILE):
    """
    Save a user correction and auto-promote patterns to rules when enough similar
    corrections accumulate (see maybe_promote_to_rule).
    """
    corrections = load_corrections(corrections_file)
    corrections.append({
        "incident_number": str(incident_number),
        "short_description": short_description,
        "description": description,
        "original_module": original_module,
        "original_type": original_type,
        "corrected_module": corrected_module,
        "corrected_type": corrected_type
    })
    promoted_rules = maybe_promote_to_rule(corrections)
    with open(corrections_file, "w", encoding="utf-8") as f:
        json.dump({"corrections": corrections, "promoted_rules": promoted_rules}, f, indent=2)


def apply_corrections(combined_text, corrections, incident_number=None):
    """
    Check if this text closely matches any previously corrected incident.
    Returns (module, type, confidence) or (None, None, 0).

    Two-pass strategy:
      1. Exact incident-number match (highest confidence, used when reprocessing known incidents).
      2. Text fuzzy match — uses token_sort_ratio (balanced, not subset-biased like token_set_ratio)
         with a higher threshold (92) and minimum description length (20 chars) to prevent
         short generic descriptions from broadly over-matching unrelated incidents.
    """
    if not corrections:
        return None, None, 0

    # Pass 1: exact incident number match — iterate in reverse so the latest correction wins
    if incident_number:
        inc_clean = str(incident_number).strip().upper()
        for corr in reversed(corrections):
            if str(corr.get("incident_number", "")).strip().upper() == inc_clean:
                return corr["corrected_module"], corr["corrected_type"], 1.0

    # Pass 2: fuzzy text match on combined correction text (short_description + description).
    # Using the full correction text means incidents with rich descriptions match more
    # accurately against past corrections that also have descriptions stored.
    for corr in reversed(corrections):
        corr_short = preprocess_text(corr.get("short_description", ""))
        corr_desc  = preprocess_text(corr.get("description", ""))
        corr_text  = f"{corr_short} {corr_desc}".strip() if corr_desc else corr_short
        if not corr_text or len(corr_text) < 20:
            continue
        score = fuzz.token_sort_ratio(combined_text, corr_text)
        if score >= 92:
            return corr["corrected_module"], corr["corrected_type"], score / 100.0

    # Pass 3: fuzzy match on description alone — lower threshold catches same-pattern incidents
    # even when short descriptions vary (e.g. "RACPAD-" vs "Unable to confirm delivery").
    for corr in reversed(corrections):
        corr_desc = preprocess_text(corr.get("description", ""))
        if not corr_desc or len(corr_desc) < 30:
            continue
        score = fuzz.token_sort_ratio(combined_text, corr_desc)
        if score >= 88:
            return corr["corrected_module"], corr["corrected_type"], score / 100.0

    return None, None, 0


def load_promoted_rules(corrections_file=CORRECTIONS_FILE):
    """
    Load auto-promoted rules from corrections.json.
    These rules are dynamically generated when 3+ corrections share the same pattern.
    """
    if os.path.exists(corrections_file):
        with open(corrections_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data.get("promoted_rules", [])
    return []


def maybe_promote_to_rule(corrections):
    """
    Analyse all corrections and auto-promote patterns to rules when 3+ corrections
    map to the same (corrected_module, corrected_type).

    For each qualifying group, common meaningful keywords are extracted from the
    incident descriptions and merged into a 'promoted rule' entry with priority 97.
    These rules sit above generic fallbacks but below hand-crafted explicit rules,
    so they act as a safety net for frequently recurring misclassifications.

    Returns the full updated promoted_rules list (to be written back to corrections.json).
    """
    # Words too generic to be useful as keywords in this domain
    STOP_WORDS = {
        "the", "a", "an", "is", "in", "to", "for", "on", "at", "of", "and", "or",
        "but", "with", "by", "from", "this", "that", "it", "be", "are", "was",
        "were", "has", "have", "had", "will", "can", "could", "would", "should",
        "racpad", "racfi", "rac", "issue", "error", "problem", "unable", "cannot",
        "need", "please", "getting", "showing", "trying", "user", "store", "when",
        "after", "just", "also", "some", "been", "they", "their", "them", "then",
        "does", "into", "over", "such", "than", "only", "about", "because", "its",
    }

    # Load existing promoted rules so we can update, not overwrite, manual ones
    promoted_map = {}  # (module, type_of_issue) -> rule dict
    if os.path.exists(CORRECTIONS_FILE):
        try:
            with open(CORRECTIONS_FILE, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            for pr in data.get("promoted_rules", []):
                key = (pr.get("module", ""), pr.get("type_of_issue", ""))
                if key[0] and key[1]:
                    promoted_map[key] = pr
        except Exception:
            pass

    # Group corrections by (corrected_module, corrected_type)
    groups = {}
    for corr in corrections:
        key = (corr.get("corrected_module", ""), corr.get("corrected_type", ""))
        if not key[0] or not key[1]:
            continue
        # Prefer full description; fall back to short_description
        text = preprocess_text(
            corr.get("description", "") or corr.get("short_description", "")
        )
        if text and len(text) > 10:
            groups.setdefault(key, []).append(text)

    MIN_CORRECTIONS = 3  # minimum corrections needed before auto-promoting

    for (module, type_), texts in groups.items():
        if len(texts) < MIN_CORRECTIONS:
            continue

        # Extract meaningful unigrams and bigrams from each description
        all_phrase_sets = []
        for text in texts:
            words = [w for w in text.split() if len(w) > 3 and w not in STOP_WORDS]
            bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
            all_phrase_sets.append(set(words + bigrams))

        # Keep phrases present in at least 60% of the correction texts
        min_count = max(2, int(len(texts) * 0.6))
        phrase_counts = Counter(
            phrase for ps in all_phrase_sets for phrase in ps
        )
        common_phrases = sorted(
            [ph for ph, cnt in phrase_counts.items() if cnt >= min_count and len(ph) > 4],
            key=lambda p: -phrase_counts[p]
        )[:20]  # cap at 20 keywords per promoted rule

        if not common_phrases:
            continue

        key = (module, type_)
        if key in promoted_map:
            # Merge new keywords into existing promoted rule
            existing_kws = set(promoted_map[key].get("keywords", []))
            promoted_map[key]["keywords"] = sorted(existing_kws | set(common_phrases))
            promoted_map[key]["source_corrections"] = len(texts)
        else:
            promoted_map[key] = {
                "keywords": common_phrases,
                "module": module,
                "type_of_issue": type_,
                "priority": 97,
                "source_corrections": len(texts),
                "description": (
                    f"Auto-promoted from {len(texts)} user corrections — "
                    f"{module} / {type_}"
                ),
            }

    return list(promoted_map.values())


# ==================== EMBEDDING CACHE ====================
def get_cache_key(historical_dir):
    """Generate a hash-based cache key from historical file modification times."""
    files_info = []
    if os.path.exists(historical_dir):
        for f in sorted(os.listdir(historical_dir)):
            if f.endswith(".xlsx"):
                filepath = os.path.join(historical_dir, f)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                files_info.append(f"{f}:{mtime}:{size}")
    return hashlib.md5("|".join(files_info).encode()).hexdigest()


def save_embeddings_cache(cache_key, vectorizer, tfidf_matrix, historical_df):
    """Save TF-IDF embeddings to disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"embeddings_{cache_key}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "df_index": historical_df.index.tolist(),
            "df_modules": historical_df["Module"].tolist(),
            "df_types": historical_df["Type Of Issue"].tolist(),
            "df_combined": historical_df["_combined"].tolist(),
        }, f)


def load_embeddings_cache(cache_key):
    """Load cached TF-IDF embeddings if available."""
    cache_path = os.path.join(CACHE_DIR, f"embeddings_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def save_models_cache(cache_key, ml_classifier, ai_reviewer):
    """Persist trained ML + AI models to disk so server restarts skip retraining."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"models_{cache_key}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"ml_classifier": ml_classifier, "ai_reviewer": ai_reviewer},
                    f, protocol=pickle.HIGHEST_PROTOCOL)


def load_models_cache(cache_key):
    """Return (ml_classifier, ai_reviewer) from disk cache, or (None, None) on miss."""
    cache_path = os.path.join(CACHE_DIR, f"models_{cache_key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["ml_classifier"], data["ai_reviewer"]
        except Exception:
            pass  # corrupted cache — fall through to rebuild
    return None, None


def register_custom_taxonomy_value(field: str, value: str) -> bool:
    """Persist a novel module or issue-type value into rules_config.json so it is
    available as a known value in future sessions.

    field : "modules" or "issue_types"
    value : the new value to register
    Returns True when the value was actually new (and written), False if it already existed.
    """
    value = value.strip()
    if not value:
        return False
    try:
        config: dict = {}
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
        existing = [v.strip().lower() for v in config.get(field, [])]
        if value.lower() in existing:
            return False
        config.setdefault(field, []).append(value)
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        # Keep the in-process VALID_* lists in sync so the current session reflects the change
        if field == "modules" and value not in VALID_MODULES:
            VALID_MODULES.append(value)
        elif field == "issue_types" and value not in VALID_ISSUE_TYPES:
            VALID_ISSUE_TYPES.append(value)
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_rules(rules_file=RULES_FILE):
    """Load rules from JSON config, sorted by priority (cached — rules rarely change)."""
    if os.path.exists(rules_file):
        with open(rules_file, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        rules = config.get("rules", [])
        # Sort by priority descending (highest priority matched first)
        rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
        return rules
    return []


def apply_rules(combined_text, rules, use_fuzzy=True):
    """
    Apply rule-based classification with priority ordering.
    Higher priority rules are checked first. Within each rule,
    longer keywords are checked first (more specific match wins).
    Falls back to fuzzy matching if exact match fails.
    Returns (module, type_of_issue, matched_keyword, priority) or (None, None, None, 0).
    """
    text = combined_text.lower()

    # Stage A: Exact keyword matching (fastest)
    for rule in rules:
        sorted_keywords = sorted(rule["keywords"], key=len, reverse=True)
        for keyword in sorted_keywords:
            kw_clean = preprocess_text(keyword)
            if kw_clean in text:
                return rule["module"], rule["type_of_issue"], keyword, rule.get("priority", 50)

    # Stage B: Fuzzy matching (for typos/variations)
    if use_fuzzy:
        for rule in rules:
            matched_kw, score = fuzzy_match_keyword(text, rule["keywords"])
            if matched_kw:
                # Reduce priority slightly for fuzzy matches
                adj_priority = max(rule.get("priority", 50) - 10, 35)
                return rule["module"], rule["type_of_issue"], f"{matched_kw} (fuzzy:{score}%)", adj_priority

    return None, None, None, 0


# ==================== HISTORICAL DATA LOADING ====================
@st.cache_data
def load_historical_data():
    """Load all historical Excel files, normalize labels, combine them."""
    all_data = []

    if not os.path.exists(HISTORICAL_DIR):
        return pd.DataFrame()

    for filename in os.listdir(HISTORICAL_DIR):
        if not filename.endswith(".xlsx"):
            continue
        filepath = os.path.join(HISTORICAL_DIR, filename)
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            try:
                df, _, _ = read_excel_robust(file_bytes)
            except ValueError:
                continue

            # Normalize column names for Type Of Issue
            col_map = {c: c for c in df.columns}
            for c in df.columns:
                if c.lower().replace(" ", "").replace("_", "") == "typeofissue":
                    col_map[c] = "Type Of Issue"
            df = df.rename(columns=col_map)

            # Only keep rows with valid Module and Type Of Issue
            if "Module" in df.columns and "Type Of Issue" in df.columns:
                valid = df.dropna(subset=["Module", "Type Of Issue"])
                if "Short Description" in valid.columns:
                    all_data.append(valid)
        except Exception:
            continue

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Normalize module names
        combined["Module"] = combined["Module"].apply(normalize_module)
        return combined
    return pd.DataFrame()


def normalize_module(module_name):
    """Normalize module name to canonical form."""
    if pd.isna(module_name):
        return "Operations"
    cleaned = str(module_name).strip().lower()
    if cleaned in MODULE_NORMALIZATION:
        return MODULE_NORMALIZATION[cleaned]
    # Title case if already valid
    titled = str(module_name).strip().title()
    if titled in VALID_MODULES:
        return titled
    return str(module_name).strip().title()


# ==================== SIMILARITY MATCHING ====================
class SimilarityMatcher:
    def __init__(self, historical_df):
        self.historical_df = historical_df
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        if self.historical_df.empty:
            return
        self.historical_df = self.historical_df.copy()
        self.historical_df["_combined"] = self.historical_df.apply(combine_text, axis=1)
        # Remove empty texts
        self.historical_df = self.historical_df[
            self.historical_df["_combined"].str.len() > 5
        ].reset_index(drop=True)

        if self.historical_df.empty:
            return

        # Try to load from cache
        cache_key = get_cache_key(HISTORICAL_DIR)
        cached = load_embeddings_cache(cache_key)

        if cached:
            self.vectorizer = cached["vectorizer"]
            self.tfidf_matrix = cached["tfidf_matrix"]
            # Restore df columns from cache
            self.historical_df = self.historical_df.head(len(cached["df_combined"]))
            self.historical_df["_combined"] = cached["df_combined"]
            self.historical_df["Module"] = cached["df_modules"]
            self.historical_df["Type Of Issue"] = cached["df_types"]
            return

        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.historical_df["_combined"]
        )

        # Save to cache
        try:
            save_embeddings_cache(cache_key, self.vectorizer, self.tfidf_matrix, self.historical_df)
        except Exception:
            pass  # Cache save failure is non-critical

    def match(self, text, threshold=None):
        """Find most similar historical record. Returns (module, type, similarity_score)."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return None, None, 0.0

        thresh = threshold if threshold is not None else SIMILARITY_THRESHOLD
        text_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(text_vec, self.tfidf_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < thresh:
            return None, None, best_score

        # Top-3 voting for more robust module and type selection
        top3_idx = np.argsort(similarities)[-3:][::-1]
        soft_thresh = thresh * 0.85
        relevant = [i for i in top3_idx if similarities[i] >= soft_thresh]

        if len(relevant) >= 2:
            mod_votes = [normalize_module(self.historical_df.iloc[i]["Module"]) for i in relevant]
            top_mod, top_mod_count = Counter(mod_votes).most_common(1)[0]
            top1_mod = normalize_module(self.historical_df.iloc[best_idx]["Module"])
            final_module = top_mod if top_mod_count >= 2 else top1_mod
            mod_winners = [i for i in relevant
                           if normalize_module(self.historical_df.iloc[i]["Module"]) == final_module]
            type_votes = [str(self.historical_df.iloc[i]["Type Of Issue"]) for i in mod_winners]
            final_type = (Counter(type_votes).most_common(1)[0][0] if type_votes
                          else str(self.historical_df.iloc[best_idx]["Type Of Issue"]))
            return final_module, final_type, best_score

        row = self.historical_df.iloc[best_idx]
        return normalize_module(row["Module"]), row["Type Of Issue"], best_score

    def batch_match(self, texts, threshold=None):
        """Batch version of match(): one matrix multiply instead of N individual calls."""
        if self.vectorizer is None or self.tfidf_matrix is None or not texts:
            return [(None, None, 0.0)] * len(texts)
        thresh = threshold if threshold is not None else SIMILARITY_THRESHOLD
        X = self.vectorizer.transform(texts)
        sims_matrix = cosine_similarity(X, self.tfidf_matrix)  # (n_texts x n_historical)
        best_indices = np.argmax(sims_matrix, axis=1)
        best_scores = sims_matrix[np.arange(len(texts)), best_indices]
        results = []
        for j in range(len(texts)):
            top_score = float(best_scores[j])
            if top_score < thresh:
                results.append((None, None, top_score))
                continue

            # Top-3 voting for more robust module and type selection
            row_sims = sims_matrix[j]
            top3_idx = np.argsort(row_sims)[-3:][::-1]
            soft_thresh = thresh * 0.85
            relevant = [i for i in top3_idx if row_sims[i] >= soft_thresh]

            if len(relevant) >= 2:
                mod_votes = [normalize_module(self.historical_df.iloc[i]["Module"])
                             for i in relevant]
                top_mod, top_mod_count = Counter(mod_votes).most_common(1)[0]
                top1_mod = normalize_module(self.historical_df.iloc[best_indices[j]]["Module"])
                final_module = top_mod if top_mod_count >= 2 else top1_mod
                # Vote on type only among neighbors that agree on the winning module
                mod_winners = [i for i in relevant
                               if normalize_module(self.historical_df.iloc[i]["Module"]) == final_module]
                type_votes = [str(self.historical_df.iloc[i]["Type Of Issue"])
                              for i in mod_winners]
                final_type = (Counter(type_votes).most_common(1)[0][0] if type_votes
                              else str(self.historical_df.iloc[best_indices[j]]["Type Of Issue"]))
                results.append((final_module, final_type, top_score))
            else:
                row = self.historical_df.iloc[best_indices[j]]
                results.append((normalize_module(row["Module"]),
                                row["Type Of Issue"], top_score))
        return results


# ==================== ML CLASSIFIER ====================
class MLClassifier:
    def __init__(self, historical_df):
        self.historical_df = historical_df
        self.vectorizer = None
        self.module_clf = None
        self.type_clf = None
        self.module_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self._trained = False
        self._train()

    def _train(self):
        if self.historical_df.empty:
            return

        df = self.historical_df.copy()
        df["_combined"] = df.apply(combine_text, axis=1)
        df = df[df["_combined"].str.len() > 5].reset_index(drop=True)

        if len(df) < 10:
            return

        # Normalize labels
        df["Module"] = df["Module"].apply(normalize_module)
        df["Type Of Issue"] = df["Type Of Issue"].str.strip()

        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )
        X = self.vectorizer.fit_transform(df["_combined"])

        # Module classifier
        y_module = self.module_encoder.fit_transform(df["Module"])
        self.module_clf = LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, C=5.0, solver="lbfgs"
        )
        self.module_clf.fit(X, y_module)

        # Type classifier
        y_type = self.type_encoder.fit_transform(df["Type Of Issue"])
        self.type_clf = LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, C=5.0, solver="lbfgs"
        )
        self.type_clf.fit(X, y_type)

        self._trained = True

    def predict(self, text):
        """Predict module and type with confidence. Returns (module, type, confidence)."""
        if not self._trained:
            return None, None, 0.0

        X = self.vectorizer.transform([text])

        # Module prediction
        module_proba = self.module_clf.predict_proba(X)[0]
        module_idx = np.argmax(module_proba)
        module_conf = module_proba[module_idx]
        module = self.module_encoder.inverse_transform([module_idx])[0]

        # Type prediction
        type_proba = self.type_clf.predict_proba(X)[0]
        type_idx = np.argmax(type_proba)
        type_conf = type_proba[type_idx]
        type_issue = self.type_encoder.inverse_transform([type_idx])[0]

        avg_conf = (module_conf + type_conf) / 2
        return module, type_issue, avg_conf

    def batch_predict(self, texts, threshold=None):
        """Batch version of predict(): one transform + one predict_proba each instead of N calls."""
        if not self._trained or not texts:
            return [(None, None, 0.0)] * len(texts)
        thresh = threshold if threshold is not None else CONFIDENCE_THRESHOLD
        X = self.vectorizer.transform(texts)
        mod_probas = self.module_clf.predict_proba(X)
        type_probas = self.type_clf.predict_proba(X)
        mod_idx = np.argmax(mod_probas, axis=1)
        type_idx = np.argmax(type_probas, axis=1)
        ai_modules = self.module_encoder.inverse_transform(mod_idx)
        ai_types = self.type_encoder.inverse_transform(type_idx)
        mod_confs = mod_probas[np.arange(len(texts)), mod_idx]
        type_confs = type_probas[np.arange(len(texts)), type_idx]
        results = []
        for j in range(len(texts)):
            conf = float((mod_confs[j] + type_confs[j]) / 2)
            if conf < thresh:
                results.append((None, None, 0.0))
            else:
                results.append((str(ai_modules[j]), str(ai_types[j]), conf))
        return results


# ==================== ISSUE TYPE INFERENCE ====================
def extract_bigpanda_app_type(short_desc_lower: str) -> str:
    """Extract the service/application name from a BigPanda short description
    and return the standard Type Of Issue string.

    Example:
        'bigpanda incident htinventorytransfer'
        → 'htinventorytransfer - Backend 500 Errors / Service Outage'
    """
    m = re.search(r"bigpanda\s+(?:incident|alert)?\s+(\S+)", short_desc_lower)
    if m:
        app = m.group(1).strip()
        return f"{app} - Backend 500 Errors / Service Outage"
    return "BigPanda - Backend 500 Errors / Service Outage"


def infer_issue_type_from_context(combined_text, module):
    """
    Module-aware issue type inference. Uses the determined `module` to constrain
    the candidate types so a Payment incident never gets a Customer-module type,
    and vice versa. Each module block returns its module-appropriate default
    when no specific pattern matches. Legacy text-only patterns are used only
    when `module` is empty/unknown.
    """
    text = combined_text.lower()
    mod = module or ""

    # ── Inventory ───────────────────────────────────────────────────────
    if mod == "Inventory":
        if any(w in text for w in ["switch out", "switchout", "pending switch out"]):
            return "Switch Out Issue"
        if any(w in text for w in ["confirm delivery", "delivery confirmation",
                                    "unable to confirm delivery"]):
            return "Confirm delivery Issue"
        if any(w in text for w in ["chargeoff", "charge off", "charged off"]):
            return "Chargeoff"
        if any(w in text for w in ["inventory transfer", "transfer the item",
                                    "item transferred to wrong store"]):
            return "Inventory Transfer Issue"
        if any(w in text for w in ["inventory status", "change the item status",
                                    "not rent ready", "rent ready",
                                    "item status change"]):
            return "Inventory Status Issue"
        if "active inventory mismatch" in text:
            return "DQ-Inventory"
        if "inventory mismatch" in text:
            return "Inventory Mismatch"
        if any(w in text for w in ["unable to receive the purchase order",
                                    "unable to receive po"]):
            return "Unable to receive the Purchase order"
        if any(w in text for w in ["unable to reverse po", "reverse purchase order"]):
            return "Unable to reverse PO"
        if "cancel the manual po" in text or "cancel manual po" in text:
            return "Cancel the Manual PO"
        if "cancel the rms po" in text or "cancel rms po" in text:
            return "Cancel the RMS PO"
        if any(w in text for w in ["purchase order", "po related"]):
            return "PO related issue"
        return "Inventory Issue"

    # ── Agreement ───────────────────────────────────────────────────────
    if mod == "Agreement":
        if any(w in text for w in ["void the agreement", "unable to void",
                                    "void agreement"]):
            return "Agreement Void Issue"
        if any(w in text for w in ["transfer the agreement", "agreement transfer"]):
            return "Agreement Transfer Issue"
        if any(w in text for w in ["unable to create agreement", "create an agreement",
                                    "agreement creation"]):
            return "Agreement Creation Issue"
        if any(w in text for w in ["agreement status", "agreement shows as active",
                                    "pif", "return the agreement"]):
            return "Agreement Status"
        if any(w in text for w in ["rac exchange", "exchange item"]):
            return "RAC exchange"
        if "active agreement mismatch" in text:
            return "DQ-Agreement"
        if "agreement delivery" in text:
            return "Agreement delivery Issue"
        return "Agreement Issue"

    # ── Payment ─────────────────────────────────────────────────────────
    if mod == "Payment":
        if any(w in text for w in ["autopay", "auto pay", "auto-pay"]):
            return "AutoPay"
        if any(w in text for w in ["payment reversal", "reverse payment",
                                    "payment getting refunded", "payment refund"]):
            return "Payment reversal"
        if any(w in text for w in ["chargeback", "charge back", "cccb"]):
            return "Chargeback"
        if "receipt reversal" in text:
            return "Receipt Reversal Issue"
        if any(w in text for w in ["receipt mismatch", "receipt not matching"]):
            return "Receipt Mismatch"
        if any(w in text for w in ["tax", "sales tax", "ldw tax"]):
            return "Tax Issue"
        if any(w in text for w in ["unable to save card", "save card details"]):
            return "Unable to Save card details"
        if "payment mode" in text:
            return "Payment mode Issue"
        if any(w in text for w in ["card payment failure", "payment not successful",
                                    "initial payment", "unable to take payment"]):
            return "Payment Error"
        if any(w in text for w in ["payment query", "payment not reflecting",
                                    "payment amount", "payment total"]):
            return "Payment Query"
        return "Payment issue"

    # ── Customer ────────────────────────────────────────────────────────
    if mod == "Customer":
        if any(w in text for w in ["merge customer", "merge the customer"]):
            return "Merge Customer"
        if "duplicate customer" in text:
            return "Duplicate customers record"
        if any(w in text for w in ["cannot search", "search issue", "no results found"]):
            return "search Issue"
        if any(w in text for w in ["login issue", "unable to link the agreement",
                                    "customer login"]):
            return "Customer Login Issue (EXT)"
        if any(w in text for w in ["customer level", "level change"]):
            return "Customer level change request"
        if "customer mismatch" in text:
            return "Customer Mismatch"
        if any(w in text for w in ["customer information", "customer details",
                                    "customer info"]):
            return "Customer information issue"
        return "Customer Issue"

    # ── Store ────────────────────────────────────────────────────────────
    if mod == "Store":
        if any(w in text for w in ["pin reset", "reset pin", "pin change"]):
            return "PIN reset"
        if any(w in text for w in ["menu access", "operations tab", "menu permission"]):
            return "Menu Permission Access"
        if any(w in text for w in ["drop reconciliation", "drop recon",
                                    "over/short", "over short", "cash drop"]):
            return "Drop reconciliation"
        if "training" in text:
            return "Training Issue"
        if "daily task" in text:
            return "Daily task"
        return "Store Access"

    # ── Reporting ────────────────────────────────────────────────────────
    if mod == "Reporting":
        if any(w in text for w in ["report discrepancy", "report discrepency"]):
            return "Report discrepancy"
        if any(w in text for w in ["daily report", "daily report mismatch"]):
            return "Daily Report Issue"
        if "account report" in text:
            return "Account Report Request"
        if any(w in text for w in ["month end", "eoy inventory", "eom"]):
            return "EOM"
        if "report access" in text:
            return "Report access"
        return "Report Request"

    # ── Operations ───────────────────────────────────────────────────────
    if mod == "Operations":
        if any(w in text for w in ["pricing issue", "incorrect pricing", "price tag"]):
            return "Pricing Issue"
        if "pricing request" in text:
            return "Pricing request"
        if any(w in text for w in ["promo", "trifecta", "promotion"]):
            return "Promo issue"
        if "promo request" in text:
            return "Promo request"
        if any(w in text for w in ["2 way sync", "two way sync", "activity log"]):
            return "2 way sync"
        if "monitoring" in text:
            return "Monitoring"
        if "duplicate ticket" in text:
            return "Duplicate ticket"
        if any(w in text for w in ["intermittent", "slow", "performance",
                                    "crash", "latency", "timeout"]):
            return "Intermittent Issue"
        return "Query"

    # ── BigPanda ─────────────────────────────────────────────────────────
    if mod == "BigPanda":
        return "BigPanda - Backend 500 Errors / Service Outage"

    # ── Menu / Container ─────────────────────────────────────────────────
    if mod == "Menu / Container":
        return "Menu Permission Access"

    # ── Account Management / Dashboard / Customer Service / Data Services ──
    if mod in ("Account Management", "Dashboard", "Customer Service", "Data Services"):
        return "Query"

    # ── Fallback: legacy text-only patterns (module empty/unknown) ──────
    if any(w in text for w in ["mismatch", "incorrect", "wrong", "discrepancy"]):
        return "Data Discrepancy"
    if any(w in text for w in ["access", "permission", "role", "unauthorized"]):
        return "Access / Permission Issue"
    if any(w in text for w in ["login", "sso", "okta", "cognito", "auth"]):
        return "Authentication / SSO Issue"
    if any(w in text for w in ["sync", "integration", "not propagat", "not reflected"]):
        return "Data Sync / Integration Failure"
    if any(w in text for w in ["config", "flag", "setting"]):
        return "Configuration Issue"
    if any(w in text for w in ["slow", "performance", "crash", "latency", "timeout"]):
        return "Intermittent Issue"

    return "One Time Issue"


# ==================== RICH PROGRESS HELPER ====================
class StageProgressBar:
    """
    Drop-in replacement for st.progress() that maps a sub-stage fraction
    to a global 0-100% master bar and shows row-level context in the label.
    UI updates are throttled to every ~1% change to avoid browser flooding.
    """
    STAGE_ICONS = ["⚙️", "🔍", "🤖", "📊"]

    def __init__(self, master_bar, status_placeholder, n_total,
                 stage_num, stage_total, stage_desc,
                 pct_start, pct_end):
        self._bar = master_bar
        self._status = status_placeholder
        self._n = n_total
        self._stage_num = stage_num
        self._stage_total = stage_total
        self._desc = stage_desc
        self._s = pct_start
        self._e = pct_end
        self._prev_pct = -1
        icon = self.STAGE_ICONS[(stage_num - 1) % len(self.STAGE_ICONS)]
        self._status.markdown(
            f"**{icon} Step {stage_num}/{stage_total} — {stage_desc}**"
        )
        self._bar.progress(pct_start / 100.0,
                           text=f"Step {stage_num}/{stage_total} — {stage_desc}")

    def progress(self, frac):
        """Accept frac 0.0–1.0 (same API as st.progress)."""
        pct = self._s + frac * (self._e - self._s)
        pct_int = int(pct)
        # Throttle: only push update when integer % changes or at completion
        if pct_int == self._prev_pct and frac < 1.0:
            return
        self._prev_pct = pct_int
        row = int(frac * self._n)
        label = (
            f"Step {self._stage_num}/{self._stage_total} — "
            f"{self._desc}: {row:,} / {self._n:,} ({pct_int}%)"
        )
        self._bar.progress(min(pct / 100.0, 1.0), text=label)


# ==================== CONFIDENCE VALIDATION ====================
def validate_confidence(module, type_of_issue, confidence, method, rule_priority=0):
    """Assign confidence level string based on method, score, and rule priority."""
    if method == "rule":
        if rule_priority >= 90:
            return "High"
        elif rule_priority >= 75:
            return "Medium"
        else:
            return "Low - Needs Review"
    elif method == "similarity":
        if confidence >= 0.92:
            return "High"
        elif confidence >= 0.85:
            return "Medium"
        else:
            return "Low - Needs Review"
    elif method == "ml":
        if confidence >= 0.75:
            return "Medium"
        elif confidence >= CONFIDENCE_THRESHOLD:
            return "Low - Needs Review"
        else:
            return "Low - Needs Review"
    return "Low - Needs Review"


# ==================== AI REVIEWER ====================
class AIReviewer:
    """
    Second-opinion AI reviewer using a calibrated LinearSVC + multi-signal voting.
    Only acts on Low/Medium confidence predictions from the primary pipeline.
    Auto-promotes to High when multiple independent signals agree above threshold.
    """
    AI_PROMOTE_THRESHOLD = 0.82

    def __init__(self, historical_df):
        self.vectorizer = None
        self.module_clf = None
        self.type_clf = None
        self.module_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self._trained = False
        self._train(historical_df)

    def _train(self, historical_df):
        if historical_df.empty or len(historical_df) < 20:
            return
        df = historical_df.copy()
        df["_combined"] = df.apply(combine_text, axis=1)
        df = df[df["_combined"].str.len() > 5].reset_index(drop=True)
        if len(df) < 10:
            return
        df["Module"] = df["Module"].apply(normalize_module)
        df["Type Of Issue"] = df["Type Of Issue"].str.strip()

        # Drop classes that have fewer than MIN_SAMPLES examples
        # so CalibratedClassifierCV(cv=2) always has ≥ 2 samples per fold
        MIN_SAMPLES = 4
        valid_mod = df["Module"].value_counts()
        valid_mod = valid_mod[valid_mod >= MIN_SAMPLES].index
        valid_type = df["Type Of Issue"].value_counts()
        valid_type = valid_type[valid_type >= MIN_SAMPLES].index
        df = df[df["Module"].isin(valid_mod) & df["Type Of Issue"].isin(valid_type)].reset_index(drop=True)
        if len(df) < 10:
            return

        self.vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True, min_df=2, max_df=0.95
        )
        X = self.vectorizer.fit_transform(df["_combined"])
        y_module = self.module_encoder.fit_transform(df["Module"])
        y_type = self.type_encoder.fit_transform(df["Type Of Issue"])

        self.module_clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=2
        )
        self.module_clf.fit(X, y_module)

        self.type_clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=2
        )
        self.type_clf.fit(X, y_type)
        self._trained = True

    def predict(self, text):
        """Returns (module, type_issue, module_conf, type_conf)."""
        if not self._trained:
            return None, None, 0.0, 0.0
        X = self.vectorizer.transform([text])
        mod_proba = self.module_clf.predict_proba(X)[0]
        mod_idx = int(np.argmax(mod_proba))
        module = self.module_encoder.inverse_transform([mod_idx])[0]
        type_proba = self.type_clf.predict_proba(X)[0]
        type_idx = int(np.argmax(type_proba))
        type_issue = self.type_encoder.inverse_transform([type_idx])[0]
        return module, type_issue, float(mod_proba[mod_idx]), float(type_proba[type_idx])

    def review(self, combined_text, current_module, current_type, similarity_matcher=None):
        """
        Multi-signal review for a Low/Medium confidence prediction.
        Returns (final_module, final_type, ai_confidence, reason, should_promote).
        """
        if not self._trained:
            return current_module, current_type, 0.0, "AI not trained", False

        ai_module_raw, ai_type_raw, mod_conf, type_conf = self.predict(combined_text)
        ai_module = normalize_module(ai_module_raw)
        ai_type = (ai_type_raw if ai_type_raw in VALID_ISSUE_TYPES
                   else infer_issue_type_from_context(combined_text, ai_module))

        base_conf = (mod_conf + type_conf) / 2
        boost = 0.0
        signals = []

        # Signal 1: module agreement with existing pipeline prediction
        if ai_module == current_module:
            boost += 0.07
            signals.append(f"agrees with existing module={ai_module}")

        # Signal 1b: type agreement with existing pipeline prediction
        if ai_type == current_type:
            boost += 0.04
            signals.append(f"agrees with existing type={ai_type}")

        # Signal 2: top-k similarity neighbour majority vote (module + type)
        if similarity_matcher and similarity_matcher.vectorizer is not None:
            try:
                text_vec = similarity_matcher.vectorizer.transform([combined_text])
                sims = cosine_similarity(text_vec, similarity_matcher.tfidf_matrix).flatten()
                top_k = np.argsort(sims)[-7:][::-1]
                relevant = [i for i in top_k if sims[i] >= 0.65]
                if len(relevant) >= 2:
                    mod_votes = [normalize_module(similarity_matcher.historical_df.iloc[i]["Module"])
                                 for i in relevant]
                    top_mod_vote, top_mod_count = Counter(mod_votes).most_common(1)[0]
                    if top_mod_vote == ai_module and top_mod_count / len(relevant) >= 0.6:
                        boost += 0.06
                        signals.append(
                            f"similarity k-NN module agrees ({top_mod_count}/{len(relevant)})"
                        )
                        # Signal 2b: kNN type majority vote (only when module vote passes)
                        type_votes = [str(similarity_matcher.historical_df.iloc[i]["Type Of Issue"])
                                      for i in relevant]
                        top_type_vote, top_type_count = Counter(type_votes).most_common(1)[0]
                        if top_type_vote == ai_type and top_type_count / len(relevant) >= 0.5:
                            boost += 0.03
                            signals.append(
                                f"similarity k-NN type agrees ({top_type_count}/{len(relevant)})"
                            )
            except Exception:
                pass

        final_conf = min(base_conf + boost, 0.97)
        signals.insert(0, f"conf={final_conf:.2f}")
        reason = "; ".join(signals)

        should_promote = final_conf >= self.AI_PROMOTE_THRESHOLD and ai_module in VALID_MODULES
        final_module = ai_module if should_promote else current_module
        final_type = ai_type if should_promote else current_type
        return final_module, final_type, final_conf, reason, should_promote

    def batch_review(self, combined_texts, current_modules, current_types,
                     similarity_matcher=None, threshold=None):
        """
        Batch version of review(): one transform + one predict_proba + one matrix multiply
        for ALL review rows instead of N individual calls.
        Returns list of (final_module, final_type, ai_conf, reason, should_promote).
        """
        if not self._trained or not combined_texts:
            return [(m, t, 0.0, "AI not trained", False)
                    for m, t in zip(current_modules, current_types)]
        thresh = threshold if threshold is not None else self.AI_PROMOTE_THRESHOLD
        n = len(combined_texts)

        # Batch AI predictions (one transform + two predict_proba calls)
        X = self.vectorizer.transform(combined_texts)
        mod_probas = self.module_clf.predict_proba(X)
        type_probas = self.type_clf.predict_proba(X)
        mod_idx = np.argmax(mod_probas, axis=1)
        type_idx = np.argmax(type_probas, axis=1)
        ai_modules_raw = self.module_encoder.inverse_transform(mod_idx)
        ai_types_raw = self.type_encoder.inverse_transform(type_idx)
        mod_confs = mod_probas[np.arange(n), mod_idx]
        type_confs = type_probas[np.arange(n), type_idx]
        base_confs = (mod_confs + type_confs) / 2.0

        # Batch similarity kNN (one transform + one matrix multiply for ALL review rows)
        sims_matrix = None
        if similarity_matcher and similarity_matcher.vectorizer is not None:
            try:
                X_sim = similarity_matcher.vectorizer.transform(combined_texts)
                sims_matrix = cosine_similarity(X_sim, similarity_matcher.tfidf_matrix)
            except Exception:
                sims_matrix = None

        results = []
        for j in range(n):
            ai_module = normalize_module(str(ai_modules_raw[j]))
            ai_type_raw = str(ai_types_raw[j])
            ai_type = (ai_type_raw if ai_type_raw in VALID_ISSUE_TYPES
                       else infer_issue_type_from_context(combined_texts[j], ai_module))

            boost = 0.0
            signals = []

            # Signal 1: module agreement with existing pipeline prediction
            if ai_module == current_modules[j]:
                boost += 0.07
                signals.append(f"agrees with existing module={ai_module}")

            # Signal 1b: type agreement with existing pipeline prediction
            if ai_type == current_types[j]:
                boost += 0.04
                signals.append(f"agrees with existing type={ai_type}")

            # Signal 2: kNN vote from pre-computed similarity matrix (module + type)
            if sims_matrix is not None:
                try:
                    sims = sims_matrix[j]
                    top_k = np.argsort(sims)[-7:][::-1]
                    relevant = [i for i in top_k if sims[i] >= 0.65]
                    if len(relevant) >= 2:
                        mod_votes = [normalize_module(similarity_matcher.historical_df.iloc[i]["Module"])
                                     for i in relevant]
                        top_mod_vote, top_mod_count = Counter(mod_votes).most_common(1)[0]
                        if top_mod_vote == ai_module and top_mod_count / len(relevant) >= 0.6:
                            boost += 0.06
                            signals.append(
                                f"similarity k-NN module agrees ({top_mod_count}/{len(relevant)})"
                            )
                            # Signal 2b: kNN type majority vote (only when module vote passes)
                            type_votes = [str(similarity_matcher.historical_df.iloc[i]["Type Of Issue"])
                                          for i in relevant]
                            top_type_vote, top_type_count = Counter(type_votes).most_common(1)[0]
                            if top_type_vote == ai_type and top_type_count / len(relevant) >= 0.5:
                                boost += 0.03
                                signals.append(
                                    f"similarity k-NN type agrees ({top_type_count}/{len(relevant)})"
                                )
                except Exception:
                    pass

            final_conf = min(float(base_confs[j]) + boost, 0.97)
            signals.insert(0, f"conf={final_conf:.2f}")
            reason = "; ".join(signals)

            should_promote = final_conf >= thresh and ai_module in VALID_MODULES
            final_module = ai_module if should_promote else current_modules[j]
            final_type = ai_type if should_promote else current_types[j]
            results.append((final_module, final_type, final_conf, reason, should_promote))

        return results


class SemanticReviewer:
    """
    Local neural AI reviewer using sentence-transformers — no API key required.
    Downloads 'all-MiniLM-L6-v2' (~90 MB) once and caches it locally.
    Reviews ALL incidents (including High confidence) for semantic plausibility.
    """
    MODEL_NAME = "all-MiniLM-L6-v2"
    OVERRIDE_THRESHOLD = 0.55   # min cosine similarity to the best module to override
    MIN_GAP_OVERRIDE = 0.12     # best module must beat current by this margin to override
    MIN_GAP_FLAG = 0.06         # flag for human review if gap is at least this

    MODULE_DESCRIPTIONS = {
        "Payment": (
            "payment processing error, payment refund, chargeback dispute, autopay enrollment "
            "or removal, payment reversal, receipt issues, tax calculation, unable to save credit card, "
            "duplicate payment, EPO payment, suspense, unbalanced receipt, payment not posting, "
            "payment declined, transaction error, payment amount incorrect, payment not applying, "
            "card payment failure, initial payment, unable to take payment, payment mode issue"
        ),
        "Agreement": (
            "rental agreement creation, void agreement, transfer agreement, agreement delivery, "
            "RAC exchange, agreement status, LDW cancellation, SAC calculation, EPO, reinstatement, "
            "rate reduction, confirm to return, agreement lookup, "
            "agreement not showing, agreement error, past due agreement, agreement modification, "
            "unable to void agreement, agreement transfer issue, agreement creation issue"
        ),
        "Inventory": (
            "inventory item management, purchase order receiving, switch out, charge off, "
            "inventory status change, inventory transfer, inventory mismatch, rent ready, "
            "item in service, loaner, PO cancellation, item not found, serialized item, "
            "item condition, inventory count, physical inventory, item lookup, "
            "unable to receive PO, purchase order issue, inventory item status"
        ),
        "Store": (
            "store access, employee login, PIN reset, authentication, drop reconciliation, "
            "store assignment, old store access, transferred store, revoke access, "
            "cash management, DAP, blank screen, store not appearing, "
            "coworker login, employee permissions, user account locked, store number login, "
            "store user access, login issue, employee account, manager access"
        ),
        "Customer": (
            "customer profile update, customer account, customer login, merge customers, "
            "duplicate customer records, customer mismatch, customer level change, benefits plus, "
            "identity verification, customer search, customer address, "
            "customer information, customer details, update customer record, customer name, "
            "customer not found, customer lookup, customer data issue"
        ),
        "Reporting": (
            "generating reports, report access, daily report, account report, report discrepancy, "
            "Power BI, collection metrics, audit trail, inventory recap, report mismatch, "
            "report not showing, report request, missing report, report generation, "
            "daily closing report, EOM report, end of month, report download"
        ),
        "Operations": (
            "system configuration, pricing issues, monitoring alerts, batch jobs, two-way sync, "
            "lambda errors, API errors, performance degradation, feature flags, translation, "
            "system outage, service degradation, application error, backend error, "
            "configuration change, promo issue, pricing request, intermittent issue"
        ),
        "Account Management": (
            "collection activities, past due accounts, route assignments, field sheets, "
            "commitment tracking, two-way text, skip stolen, extension, delinquent account, "
            "past due balance, collection agent, field collection, RPP, delinquent customer, "
            "skip trace, inbound collection, collection route, debt collection, past due"
        ),
        "Dashboard": (
            "dashboard home screen display, store widgets, action items, store events, "
            "racpad home page not loading, dashboard widget missing, "
            "home page error, welcome screen, dashboard loading issue, home screen blank, "
            "dashboard not loading, action item missing, home page blank"
        ),
        "Customer Service": (
            "customer service tickets, CSM module, medallia survey, queries, servicenow ticket, "
            "customer complaint handling, customer service request, CSM case, support ticket, "
            "customer feedback, escalation, customer call, inbound inquiry"
        ),
        "Data Services": (
            "database fixes, data synchronization, data quality issues, SQL query, oracle database, "
            "racdb, entdb, data mismatch, DQ issue, database error, data repair, data correction, "
            "database query, SQL fix, data cleanup, active inventory mismatch, active agreement mismatch"
        ),
        "BigPanda": (
            "BigPanda automated monitoring alert, system alert notification, automated incident, "
            "infrastructure alert, server alert, automated monitoring, system health alert, "
            "BigPanda ticket, monitoring notification"
        ),
        "Menu / Container": (
            "application menu, navigation bar, container shell, app shell, menu item missing, "
            "menu permission, navigation issue, app navigation, menu not showing, UI menu, "
            "menu access issue, container not loading, navigation bar missing"
        ),
    }

    def __init__(self):
        self._available = False
        self._model = None
        self._module_names = []
        self._module_emb_matrix = None
        self._init_error = ""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as _np

            # Try downloading with normal SSL verification first.
            # If the environment has a corporate proxy that intercepts HTTPS
            # (e.g. Infosys network), the SSL handshake fails and we retry with
            # verify=False. In a normal VDI or home network the first attempt
            # succeeds and SSL verification is never disabled.
            # After the ~90 MB model is cached locally, no network calls occur.
            try:
                self._model = SentenceTransformer(self.MODEL_NAME)
            except Exception as _ssl_err:
                _err_str = str(_ssl_err).lower()
                if "ssl" in _err_str or "certificate" in _err_str or "connect" in _err_str:
                    # SSL failure — likely a corporate proxy. Retry with verify=False.
                    try:
                        import httpx as _httpx
                        from huggingface_hub.utils._http import (
                            set_client_factory as _set_factory,
                            close_session as _close_session,
                        )

                        def _no_verify_client() -> _httpx.Client:
                            return _httpx.Client(verify=False)

                        _set_factory(_no_verify_client)
                        _close_session()
                    except Exception:
                        pass
                    self._model = SentenceTransformer(self.MODEL_NAME)
                else:
                    raise  # non-SSL error — propagate normally
            descriptions = list(self.MODULE_DESCRIPTIONS.values())
            self._module_names = list(self.MODULE_DESCRIPTIONS.keys())
            embs = self._model.encode(
                descriptions, convert_to_numpy=True, show_progress_bar=False
            )
            norms = _np.linalg.norm(embs, axis=1, keepdims=True)
            self._module_emb_matrix = embs / _np.maximum(norms, 1e-9)
            self._available = True
        except ImportError:
            self._init_error = "sentence-transformers not installed — run: pip install sentence-transformers"
        except Exception as _e:
            self._init_error = str(_e)

    def batch_review(self, combined_texts, current_modules, current_types, current_confidences):
        """
        Review a batch of incidents semantically.
        Returns list of dicts: final_module, final_type, semantic_conf,
        semantic_status ("Overridden" | "Flagged → <module>" | "Confirmed"), semantic_reason.
        """
        import numpy as _np

        n = len(combined_texts)
        fallback = [
            {
                "final_module": current_modules[i],
                "final_type": current_types[i],
                "semantic_conf": 0.0,
                "semantic_status": "Unavailable",
                "semantic_reason": "sentence-transformers not installed",
            }
            for i in range(n)
        ]
        if not self._available or self._model is None:
            return fallback

        try:
            raw_embs = self._model.encode(
                combined_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )
            norms = _np.linalg.norm(raw_embs, axis=1, keepdims=True)
            inc_embs = raw_embs / _np.maximum(norms, 1e-9)

            # sim_matrix[i, j] = cosine similarity of incident i vs module description j
            sim_matrix = inc_embs @ self._module_emb_matrix.T  # shape (n, n_modules)

            results = []
            for i in range(n):
                sims = sim_matrix[i]
                best_idx = int(_np.argmax(sims))
                best_module = self._module_names[best_idx]
                best_score = float(sims[best_idx])

                curr_module = current_modules[i]
                curr_type = current_types[i]
                curr_conf = current_confidences[i]

                if curr_module in self._module_names:
                    curr_idx = self._module_names.index(curr_module)
                    curr_score = float(sims[curr_idx])
                else:
                    curr_score = 0.0

                score_gap = best_score - curr_score

                # Require a larger gap to override High-confidence classifications
                gap_needed = (
                    self.MIN_GAP_OVERRIDE * 1.5
                    if curr_conf == "High"
                    else self.MIN_GAP_OVERRIDE
                )

                if (best_module != curr_module
                        and best_score >= self.OVERRIDE_THRESHOLD
                        and score_gap >= gap_needed):
                    status = "Overridden"
                    reason = (
                        f"Semantic model strongly suggests '{best_module}' "
                        f"(score={best_score:.3f}) over '{curr_module}' "
                        f"(score={curr_score:.3f}, gap={score_gap:.3f})"
                    )
                    final_module = best_module
                    final_type = curr_type
                elif (best_module != curr_module and score_gap >= self.MIN_GAP_FLAG):
                    status = f"Flagged → {best_module}"
                    reason = (
                        f"Semantic model prefers '{best_module}' "
                        f"(score={best_score:.3f}) over '{curr_module}' "
                        f"(score={curr_score:.3f}, gap={score_gap:.3f})"
                    )
                    final_module = curr_module
                    final_type = curr_type
                else:
                    status = "Confirmed"
                    reason = (
                        f"Semantic model agrees with '{curr_module}' "
                        f"(score={curr_score:.3f}, best_gap={score_gap:.3f})"
                    )
                    final_module = curr_module
                    final_type = curr_type

                results.append({
                    "final_module": final_module,
                    "final_type": final_type,
                    "semantic_conf": round(best_score, 3),
                    "semantic_status": status,
                    "semantic_reason": reason,
                })
            return results

        except Exception as exc:
            for r in fallback:
                r["semantic_reason"] = f"Error during semantic review: {exc}"
            return fallback


def apply_ai_review(classification_df, input_df, ai_reviewer, similarity_matcher,
                    auto_promote_threshold=0.82):
    """
    Post-pipeline AI review pass for Medium/Low confidence incidents.
    Uses batch_review() — one vectorize + predict pass over ALL review rows
    instead of N individual per-row calls.
    Returns (updated_classification_df, promoted_count).
    """
    result = classification_df.copy()
    result["AI Review"] = "Skipped"
    result["AI Confidence"] = 0.0
    result["AI Reason"] = ""

    if not ai_reviewer or not ai_reviewer._trained:
        return result, 0

    review_mask = result["Confidence"].isin(["Low - Needs Review", "Medium"])
    review_positions = [i for i in range(len(result)) if review_mask.iloc[i]]

    if not review_positions:
        return result, 0

    input_reset = input_df.reset_index(drop=True)
    mod_col = result.columns.get_loc("Module")
    type_col = result.columns.get_loc("Type Of Issue")
    conf_col = result.columns.get_loc("Confidence")
    method_col = result.columns.get_loc("Classification Method")
    ai_rev_col = result.columns.get_loc("AI Review")
    ai_conf_col = result.columns.get_loc("AI Confidence")
    ai_reason_col = result.columns.get_loc("AI Reason")

    # Batch: collect ALL review row texts + current labels at once
    combined_texts = [combine_text(input_reset.iloc[pos]) for pos in review_positions]
    current_modules = [result.iat[pos, mod_col] for pos in review_positions]
    current_types = [result.iat[pos, type_col] for pos in review_positions]
    current_methods = [result.iat[pos, method_col] for pos in review_positions]

    # One batch call instead of N individual review() calls
    batch_reviews = ai_reviewer.batch_review(
        combined_texts, current_modules, current_types,
        similarity_matcher, threshold=auto_promote_threshold
    )

    promoted = 0
    for j, pos in enumerate(review_positions):
        final_module, final_type, ai_conf, reason, should_promote = batch_reviews[j]
        result.iat[pos, ai_conf_col] = round(ai_conf, 3)
        result.iat[pos, ai_reason_col] = reason
        if should_promote and ai_conf >= auto_promote_threshold:
            result.iat[pos, mod_col] = final_module
            result.iat[pos, type_col] = final_type
            result.iat[pos, conf_col] = "High"
            result.iat[pos, method_col] = current_methods[j] + f" → AI ({ai_conf:.2f})"
            result.iat[pos, ai_rev_col] = "Promoted"
            promoted += 1
        else:
            result.iat[pos, ai_rev_col] = "Reviewed"

    return result, promoted


def apply_semantic_review(classification_df, input_df, semantic_reviewer):
    """
    Neural AI second-opinion pass over ALL incidents (including High confidence).
    Adds columns: "Semantic Review", "Semantic Confidence", "Semantic Reason".
    Overridden rows get Module/Confidence updated and Classification Method appended.
    Returns (updated_df, override_count, flag_count).
    """
    result = classification_df.copy()
    result["Semantic Review"] = "Skipped"
    result["Semantic Confidence"] = 0.0
    result["Semantic Reason"] = ""

    if not semantic_reviewer or not semantic_reviewer._available:
        return result, 0, 0

    input_reset = input_df.reset_index(drop=True)
    n = len(result)

    mod_col = result.columns.get_loc("Module")
    type_col = result.columns.get_loc("Type Of Issue")
    conf_col = result.columns.get_loc("Confidence")
    method_col = result.columns.get_loc("Classification Method")
    sem_rev_col = result.columns.get_loc("Semantic Review")
    sem_conf_col = result.columns.get_loc("Semantic Confidence")
    sem_reason_col = result.columns.get_loc("Semantic Reason")

    combined_texts = [combine_text(input_reset.iloc[i]) for i in range(n)]
    current_modules = [result.iat[i, mod_col] for i in range(n)]
    current_types = [result.iat[i, type_col] for i in range(n)]
    current_confidences = [result.iat[i, conf_col] for i in range(n)]

    reviews = semantic_reviewer.batch_review(
        combined_texts, current_modules, current_types, current_confidences
    )

    override_count = 0
    flag_count = 0
    for i, rev in enumerate(reviews):
        result.iat[i, sem_conf_col] = rev["semantic_conf"]
        result.iat[i, sem_reason_col] = rev["semantic_reason"]
        status = rev["semantic_status"]

        if status == "Overridden":
            new_module = rev["final_module"]
            # Re-infer type for the new module — the old type may belong to the wrong module
            new_type = infer_issue_type_from_context(combined_texts[i], new_module)
            result.iat[i, mod_col] = new_module
            result.iat[i, type_col] = new_type
            result.iat[i, conf_col] = "High"
            result.iat[i, method_col] = (
                result.iat[i, method_col]
                + f" → Semantic ({rev['semantic_conf']:.2f})"
            )
            result.iat[i, sem_rev_col] = "Overridden"
            override_count += 1
        elif status.startswith("Flagged"):
            result.iat[i, sem_rev_col] = status
            flag_count += 1
        else:
            result.iat[i, sem_rev_col] = "Confirmed"

    return result, override_count, flag_count


# ==================== MAIN CLASSIFICATION PIPELINE ====================
def classify_incidents(df, rules, similarity_matcher, ml_classifier, progress_bar=None,
                       sim_threshold=None, conf_threshold=None):
    """
    Multi-stage hybrid classification pipeline.
    Priority: Corrections > Rules (exact+fuzzy) > Similarity > ML > Issue Type Inference

    sim_threshold  : override SIMILARITY_THRESHOLD (from sidebar slider).
    conf_threshold : override CONFIDENCE_THRESHOLD for ML (from sidebar slider).
    Performance: Keywords are pre-processed once (not N×275×10 times), then
    similarity and ML calls are batched as single matrix operations.
    """
    total = len(df)
    corrections = load_corrections()
    promoted_rules = load_promoted_rules()
    synonym_config = load_synonyms()

    # ── Pre-compute all texts once (avoid recomputing per stage) ────────────────
    combined_texts = [combine_text(row) for _, row in df.iterrows()]
    expanded_texts = [expand_text_with_synonyms(t, synonym_config) for t in combined_texts]

    # Pre-compute short descriptions (lowercased) for BigPanda fast-path detection
    short_descs_lower = []
    inc_numbers = []
    inc_col = next((c for c in ["Number", "Incident Number", "incident_number", "INC#", "INC"]
                    if c in df.columns), None)
    for _, row in df.iterrows():
        short_descs_lower.append(preprocess_text(row.get("Short Description", "")))
        inc_numbers.append(str(row[inc_col]).strip() if inc_col and pd.notna(row[inc_col]) else "")

    # ── Pre-process rule keywords once (sort + preprocess outside the row loop) ─
    # Merge static rules with auto-promoted rules (promoted_rules have priority 97,
    # so they sit above generic fallbacks but below explicit hand-crafted rules).
    all_rules = sorted(rules + promoted_rules, key=lambda r: r.get("priority", 0), reverse=True)
    fast_rules = []
    for rule in all_rules:  # merged static + auto-promoted, sorted by priority
        kws = sorted(
            [(preprocess_text(kw), kw) for kw in rule.get("keywords", [])],
            key=lambda x: -len(x[0])
        )
        fast_rules.append((
            rule["module"],
            rule["type_of_issue"],
            rule.get("priority", 50),
            kws,                        # (clean, original) pairs for exact match
            rule.get("keywords", [])    # originals kept for fuzzy fallback
        ))

    # ── Pass 1: Corrections + Rules (per-row, fast) ──────────────────────────────
    results = [None] * total
    pending_sim = []  # indices not resolved by corrections or rules

    for i, (combined, expanded) in enumerate(zip(combined_texts, expanded_texts)):
        text_lower = expanded.lower()

        # Stage -1: BigPanda fast-path — if "bigpanda" appears in the Short Description,
        # classify immediately without consulting corrections, rules, or ML.
        # Necessary because BigPanda incidents often mention "inventory transfer" in their
        # description, which causes higher-priority inventory rules to fire first.
        if "bigpanda" in short_descs_lower[i]:
            results[i] = {
                "Module": "BigPanda",
                "Type Of Issue": extract_bigpanda_app_type(short_descs_lower[i]),
                "Confidence": "High",
                "Classification Method": "rule (bigpanda short description)"
            }
            if progress_bar:
                progress_bar.progress((i + 1) / total * 0.5)
            continue

        # Stage 0: corrections (learned from user feedback)
        corr_module, corr_type, corr_score = apply_corrections(combined, corrections, inc_numbers[i])
        if corr_module and corr_type:
            conf_level = "High" if corr_score >= 0.95 else "Medium"
            results[i] = {
                "Module": corr_module,
                "Type Of Issue": corr_type,
                "Confidence": conf_level,
                "Classification Method": f"learned ({corr_score:.2f})"
            }
            if progress_bar:
                progress_bar.progress((i + 1) / total * 0.5)
            continue

        # Stage 1: exact rule match (keywords pre-processed — no overhead per row)
        matched = False
        for mod, typ, pri, kws, orig_kws in fast_rules:
            for kw_clean, kw_orig in kws:
                if kw_clean in text_lower:
                    mod_norm = normalize_module(mod)
                    conf_level = validate_confidence(mod_norm, typ, 1.0, "rule", pri)
                    results[i] = {
                        "Module": mod_norm,
                        "Type Of Issue": typ,
                        "Confidence": conf_level,
                        "Classification Method": f"rule ({kw_orig})"
                    }
                    matched = True
                    break
            if matched:
                break

        if matched:
            if progress_bar:
                progress_bar.progress((i + 1) / total * 0.5)
            continue

        # Stage 1b: fuzzy fallback (rare — only fires when exact matching fails)
        for mod, typ, pri, kws, orig_kws in fast_rules:
            fuzz_kw, score = fuzzy_match_keyword(expanded, orig_kws)
            if fuzz_kw:
                adj_pri = max(pri - 10, 35)
                mod_norm = normalize_module(mod)
                conf_level = validate_confidence(mod_norm, typ, 1.0, "rule", adj_pri)
                results[i] = {
                    "Module": mod_norm,
                    "Type Of Issue": typ,
                    "Confidence": conf_level,
                    "Classification Method": f"rule ({fuzz_kw} (fuzzy:{score}%)"
                }
                matched = True
                break

        if matched:
            if progress_bar:
                progress_bar.progress((i + 1) / total * 0.5)
            continue

        # Not resolved — queue for batch stages
        pending_sim.append(i)
        if progress_bar:
            progress_bar.progress((i + 1) / total * 0.5)

    # ── Pass 2: Batch similarity for all pending rows ────────────────────────────
    # One vectorizer.transform(n) + one cosine_similarity(n × 9822) matrix multiply.
    pending_ml = []
    if pending_sim:
        batch_texts = [combined_texts[i] for i in pending_sim]
        sim_batch = similarity_matcher.batch_match(batch_texts, threshold=sim_threshold)
        for j, i in enumerate(pending_sim):
            sim_module, sim_type, sim_score = sim_batch[j]
            if sim_module and sim_type:
                sim_module = normalize_module(sim_module)
                norm_type = (sim_type if sim_type in VALID_ISSUE_TYPES
                             else infer_issue_type_from_context(combined_texts[i], sim_module))
                conf_level = validate_confidence(sim_module, norm_type, sim_score, "similarity", 0)
                results[i] = {
                    "Module": sim_module,
                    "Type Of Issue": norm_type,
                    "Confidence": conf_level,
                    "Classification Method": f"similarity ({sim_score:.2f})"
                }
            else:
                pending_ml.append(i)

    if progress_bar:
        progress_bar.progress(0.75)

    # ── Pass 3: Batch ML for remaining rows ──────────────────────────────────────
    # One transform + two predict_proba calls for all remaining rows.
    if pending_ml:
        batch_texts = [combined_texts[i] for i in pending_ml]
        ml_batch = ml_classifier.batch_predict(batch_texts, threshold=conf_threshold)
        for j, i in enumerate(pending_ml):
            ml_module, ml_type, ml_conf = ml_batch[j]
            if ml_module and ml_type:
                ml_module = normalize_module(ml_module)
                norm_type = (ml_type if ml_type in VALID_ISSUE_TYPES
                             else infer_issue_type_from_context(combined_texts[i], ml_module))
                conf_level = validate_confidence(ml_module, norm_type, ml_conf, "ml", 0)
                results[i] = {
                    "Module": ml_module,
                    "Type Of Issue": norm_type,
                    "Confidence": conf_level,
                    "Classification Method": f"ml ({ml_conf:.2f})"
                }
            else:
                results[i] = {
                    "Module": "Unknown",
                    "Type Of Issue": "Needs Review",
                    "Confidence": "Low - Needs Review",
                    "Classification Method": "none"
                }

    # Fill any remaining None (safety net)
    for i in range(total):
        if results[i] is None:
            results[i] = {
                "Module": "Unknown",
                "Type Of Issue": "Needs Review",
                "Confidence": "Low - Needs Review",
                "Classification Method": "none"
            }

    # ── BigPanda post-process ────────────────────────────────────────────────────
    # Any BigPanda incident classified via rules/similarity/ML keeps "BigPanda" as
    # the Type Of Issue (from rules_config). Re-derive the descriptive type from
    # the short description where possible.
    for i in range(total):
        if (results[i].get("Module") == "BigPanda"
                and results[i].get("Type Of Issue") == "BigPanda"):
            results[i]["Type Of Issue"] = extract_bigpanda_app_type(short_descs_lower[i])

    if progress_bar:
        progress_bar.progress(1.0)

    return pd.DataFrame(results)


# ==================== SUMMARY GENERATION ====================
def generate_summary(df):
    """Group by Type Of Issue and count."""
    summary = df.groupby("Type Of Issue").size().reset_index(name="Count")
    summary = summary.sort_values("Count", ascending=False).reset_index(drop=True)
    return summary


def generate_module_summary(df):
    """Group by Module and count."""
    summary = df.groupby("Module").size().reset_index(name="Count")
    summary = summary.sort_values("Count", ascending=False).reset_index(drop=True)
    return summary


def generate_cross_summary(df):
    """Cross-tabulation: Module vs Type Of Issue."""
    cross = pd.crosstab(df["Module"], df["Type Of Issue"], margins=True, margins_name="Total")
    return cross


# ==================== EXCEL EXPORT ====================
def export_excel(original_df, classification_df, summary_df, module_summary_df):
    """Export to Excel with multiple sheets."""
    output = BytesIO()

    # Merge classification results into original
    result_df = original_df.copy()
    result_df["Module"] = classification_df["Module"].values
    result_df["Type Of Issue"] = classification_df["Type Of Issue"].values
    result_df["Confidence"] = classification_df["Confidence"].values
    result_df["Classification Method"] = classification_df["Classification Method"].values
    # Include AI review columns if present
    for ai_col in ["AI Review", "AI Confidence", "AI Reason"]:
        if ai_col in classification_df.columns:
            result_df[ai_col] = classification_df[ai_col].values
    # Include Semantic review columns if present
    for sem_col in ["Semantic Review", "Semantic Confidence", "Semantic Reason"]:
        if sem_col in classification_df.columns:
            result_df[sem_col] = classification_df[sem_col].values

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Updated Dump", index=False)
        summary_df.to_excel(writer, sheet_name="Issue Type Summary", index=False)
        module_summary_df.to_excel(writer, sheet_name="Module Summary", index=False)
        # Cross summary
        cross = generate_cross_summary(classification_df)
        cross.to_excel(writer, sheet_name="Module vs Issue Type")

    output.seek(0)
    return output, result_df


@st.cache_resource(show_spinner=False)
def build_semantic_reviewer():
    """Load sentence-transformers model for semantic review. Downloads ~90 MB on first run."""
    return SemanticReviewer()


# ==================== CACHED MODEL BUILDER ====================
@st.cache_resource(show_spinner=False)
def build_classification_models():
    """
    Load or build all models. On first run, trains and saves to disk (~5 min).
    Every subsequent restart loads from disk in seconds — no retraining needed
    unless historical files change (cache key is based on file mtimes/sizes).
    """
    hist_df = load_historical_data()
    cache_key = get_cache_key(HISTORICAL_DIR)

    # SimilarityMatcher manages its own embedding cache internally
    similarity_matcher = SimilarityMatcher(hist_df)

    # Try loading ML + AI models from disk first
    ml_classifier, ai_reviewer = load_models_cache(cache_key)
    if ml_classifier is None or ai_reviewer is None:
        # Cache miss — train from scratch then persist to disk
        ml_classifier = MLClassifier(hist_df)
        ai_reviewer = AIReviewer(hist_df)
        try:
            save_models_cache(cache_key, ml_classifier, ai_reviewer)
        except Exception:
            pass  # cache save failure is non-critical

    return similarity_matcher, ml_classifier, ai_reviewer


# ==================== INLINE CORRECTION HELPERS ====================

def _regen_excel_bytes():
    """Regenerate and cache the Excel download bytes from current session state."""
    cdf = st.session_state["classification_df"]
    odf = st.session_state["original_df"]
    sdf = generate_summary(cdf)
    mdf = generate_module_summary(cdf)
    excel_output, _ = export_excel(odf, cdf, sdf, mdf)
    st.session_state["excel_bytes"] = excel_output.getvalue()


def _apply_inline_corrections(original_slice, edited_slice, extra_updates=None, confirm_col=None):
    """Compare original and edited DataFrames, save corrections for changed rows,
    and update session state. Returns the number of rows processed (corrected or confirmed).

    extra_updates : optional dict of {col: value} applied to every processed row.
    confirm_col   : name of a boolean column in edited_slice; when True the row is
                    treated as confirmed-correct even if Module/Type Of Issue are unchanged.
    """
    result_df = st.session_state["result_df"].copy()
    classification_df = st.session_state["classification_df"].copy()
    changes = 0

    for i in range(len(original_slice)):
        orig_row = original_slice.iloc[i]
        edit_row = edited_slice.iloc[i]

        orig_module = str(orig_row.get("Module", "") or "").strip()
        edit_module = str(edit_row.get("Module", "") or "").strip()
        orig_type   = str(orig_row.get("Type Of Issue", "") or "").strip()
        edit_type   = str(edit_row.get("Type Of Issue", "") or "").strip()

        # Guard: skip rows where the user cleared a required field
        if not edit_module or not edit_type:
            continue

        module_changed = (orig_module != edit_module) or (orig_type != edit_type)
        confirmed      = confirm_col is not None and bool(edit_row.get(confirm_col, False))

        if not module_changed and not confirmed:
            continue

        number       = str(orig_row.get("Number", "") or "")
        short_desc   = str(orig_row.get("Short Description", "") or "")
        description_text = str(orig_row.get("Description", "") or "")

        # Auto-register any brand-new module / issue-type into rules_config.json
        if module_changed:
            register_custom_taxonomy_value("modules", edit_module)
            register_custom_taxonomy_value("issue_types", edit_type)

        # For confirmed-correct rows use the same original values; for edits use new values
        save_module = edit_module if module_changed else orig_module
        save_type   = edit_type   if module_changed else orig_type
        save_correction(number, orig_module, orig_type, save_module, save_type, short_desc,
                        description_text)

        if "Number" in result_df.columns and number:
            mask = result_df["Number"].astype(str) == number
            if module_changed:
                result_df.loc[mask, "Module"]        = edit_module
                result_df.loc[mask, "Type Of Issue"] = edit_type
            if extra_updates:
                for col, val in extra_updates.items():
                    if col in result_df.columns:
                        result_df.loc[mask, col] = val
            for pos in result_df[mask].index:
                if pos in classification_df.index:
                    if module_changed:
                        classification_df.loc[pos, "Module"]        = edit_module
                        classification_df.loc[pos, "Type Of Issue"] = edit_type
                    if extra_updates:
                        for col, val in extra_updates.items():
                            if col in classification_df.columns:
                                classification_df.loc[pos, col] = val

        changes += 1

    if changes > 0:
        st.session_state["result_df"]         = result_df
        st.session_state["classification_df"] = classification_df
        _regen_excel_bytes()

    return changes


# ==================== STREAMLIT UI ====================
def _render_comparison_tab():
    """Render the Trend Comparison tab — compare 2-3 already-classified dumps."""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PALETTE = px.colors.qualitative.Set2

    def _horiz_bar(wide_df, cat_col, title, top_n=None):
        """Convert a wide (categories × periods) DataFrame to a horizontal grouped bar chart."""
        _df = wide_df.copy()
        if top_n:
            _df = _df.head(top_n)
        # sort ascending so largest bar is at top of chart
        _df = _df.sort_values(_df.columns[0], ascending=True)
        _long = (
            _df.reset_index()
            .rename(columns={"index": cat_col})
            .melt(id_vars=cat_col, var_name="Period", value_name="Count")
        )
        _fig = px.bar(
            _long,
            x="Count", y=cat_col, color="Period",
            barmode="group", orientation="h",
            title=title,
            color_discrete_sequence=_PALETTE,
            height=max(380, len(_df) * 38),
            text="Count",
        )
        _fig.update_traces(textposition="outside", textfont_size=11)
        _fig.update_layout(
            yaxis_title="",
            xaxis_title="Incident Count",
            legend_title="Period",
            margin=dict(l=10, r=30, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        _fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        return _fig

    st.subheader("\U0001f4ca Monthly Incident Trend Comparison")
    st.markdown(
        "Upload **2 or 3 already-classified** Excel files (output from the "
        "Classification Tool) to compare incident trends across time periods."
    )

    # ── File uploaders ───────────────────────────────────────────────────────
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        _lbl1  = st.text_input("Period 1 label", "Month 1", key="cmp_label1")
        _file1 = st.file_uploader("Upload Period 1 (.xlsx)", type=["xlsx"], key="cmp_file1")
    with _c2:
        _lbl2  = st.text_input("Period 2 label", "Month 2", key="cmp_label2")
        _file2 = st.file_uploader("Upload Period 2 (.xlsx)", type=["xlsx"], key="cmp_file2")
    with _c3:
        _lbl3  = st.text_input("Period 3 label (optional)", "Month 3", key="cmp_label3")
        _file3 = st.file_uploader("Upload Period 3 — optional (.xlsx)", type=["xlsx"], key="cmp_file3")

    # ── Load & validate files ────────────────────────────────────────────────
    dfs = {}
    for _lbl, _f in [(_lbl1, _file1), (_lbl2, _file2), (_lbl3, _file3)]:
        if _f is None:
            continue
        try:
            _fb  = _f.read()
            _df, _sheet, _sheets = read_excel_robust(_fb)
            if "Classified Results" in _sheets:
                _df, _, _ = read_excel_robust(_fb, sheet_name="Classified Results")
            if "Module" not in _df.columns or "Type Of Issue" not in _df.columns:
                st.warning(
                    f"\u26a0\ufe0f **{_lbl}**: `Module` / `Type Of Issue` columns not found — "
                    "is this a classified output file from the Classification Tool?"
                )
            else:
                # Normalise case so the same value with different casing
                # isn't counted as two separate categories
                for _col in ("Module", "Type Of Issue"):
                    if _col in _df.columns:
                        _df[_col] = _df[_col].astype(str).str.strip().str.title()
                dfs[_lbl] = _df
        except Exception as _ex:
            st.error(f"Error reading **{_lbl}**: {_ex}")

    # Duplicate-label guard: if two uploads share the same period label the second
    # silently overwrites the first in the dict, leaving only one entry.
    if len(dfs) < 2:
        if sum(1 for _, _f in [(_lbl1, _file1), (_lbl2, _file2), (_lbl3, _file3)] if _f is not None) >= 2:
            st.error(
                "\u26a0\ufe0f **Period labels must be unique.** "
                "Two or more uploads share the same label — please give each period a different name."
            )
        else:
            st.info("\U0001f446 Upload at least **2** classified incident files to start comparing.")
        return

    _labels = list(dfs.keys())

    # ── Summary metrics row ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### \U0001f4c8 Summary Metrics")
    _m_cols = st.columns(len(dfs))
    for _i, (_lbl, _df) in enumerate(dfs.items()):
        with _m_cols[_i]:
            st.metric(f"**{_lbl}** — Total Incidents", f"{len(_df):,}")
            if "Confidence" in _df.columns:
                _n_hi  = int((_df["Confidence"] == "High").sum())
                _n_rev = int(_df["Confidence"].isin(["Low - Needs Review", "Medium"]).sum())
                st.metric("High Confidence", _n_hi)
                st.metric("Needs Review",    _n_rev)

    # ── Module distribution ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### \U0001f3f7\ufe0f Module Distribution")
    _mod_data = {_lbl: _df["Module"].value_counts() for _lbl, _df in dfs.items()}
    _mod_df   = (
        pd.DataFrame(_mod_data)
        .fillna(0).astype(int)
        .sort_values(_labels[0], ascending=False)
    )
    # Normalise case: merge rows that differ only in capitalisation
    _mod_df.index = _mod_df.index.str.strip().str.title()
    _mod_df = _mod_df.groupby(_mod_df.index).sum().sort_values(_labels[0], ascending=False)
    st.plotly_chart(
        _horiz_bar(_mod_df, "Module", "Module Distribution"),
        use_container_width=True,
    )
    with st.expander("Module counts — full table"):
        st.dataframe(_mod_df, use_container_width=True)

    # ── Type of Issue distribution (top 20) ──────────────────────────────────
    st.markdown("---")
    st.markdown("### \U0001f516 Type of Issue Distribution (Top 20)")
    _type_data = {_lbl: _df["Type Of Issue"].value_counts() for _lbl, _df in dfs.items()}
    _type_df   = (
        pd.DataFrame(_type_data)
        .fillna(0).astype(int)
    )
    # Merge any residual duplicates that differ only in capitalisation
    _type_df.index = _type_df.index.str.strip().str.title()
    _type_df = _type_df.groupby(_type_df.index).sum().sort_values(_labels[0], ascending=False)
    st.plotly_chart(
        _horiz_bar(_type_df, "Type Of Issue", "Type of Issue Distribution (Top 20)", top_n=20),
        use_container_width=True,
    )
    with st.expander("Type Of Issue counts — full table"):
        st.dataframe(_type_df, use_container_width=True)

    # ── Module drill-down ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### \U0001f50d Module Drill-down — Issue Types per Module")
    _all_mods = sorted(
        set().union(*[set(_df["Module"].dropna().str.strip().str.title()) for _df in dfs.values()])
    )
    _sel_mod = st.selectbox("Select Module", _all_mods, key="cmp_mod_select")
    if _sel_mod:
        _drill = {}
        for _lbl, _df in dfs.items():
            _mask = _df["Module"].str.strip().str.title() == _sel_mod
            _drill[_lbl] = _df[_mask]["Type Of Issue"].value_counts()
        _drill_df = (
            pd.DataFrame(_drill)
            .fillna(0).astype(int)
        )
        # Sort by total across all periods so the chart is useful even when
        # the first period has 0 incidents for the selected module.
        _drill_df = _drill_df.loc[_drill_df.sum(axis=1).sort_values(ascending=False).index]
        st.plotly_chart(
            _horiz_bar(_drill_df, "Type Of Issue", f"Issue Types — {_sel_mod}"),
            use_container_width=True,
        )
        with st.expander(f"Type Of Issue for \u2018{_sel_mod}\u2019 — full table"):
            st.dataframe(_drill_df, use_container_width=True)

    # ── Period-over-Period delta (diverging bar) ─────────────────────────────
    st.markdown("---")
    st.markdown(
        f"### \U0001f4c9 Period-over-Period Change: "
        f"**{_labels[0]}** \u2192 **{_labels[1]}**"
    )
    _mc0 = dfs[_labels[0]]["Module"].value_counts()
    _mc1 = dfs[_labels[1]]["Module"].value_counts()
    _delta = pd.DataFrame({_labels[0]: _mc0, _labels[1]: _mc1}).fillna(0).astype(int)
    _delta["Change"] = _delta[_labels[1]] - _delta[_labels[0]]
    _delta["Change %"] = (
        (_delta["Change"] / _delta[_labels[0]].replace(0, 1)) * 100
    ).round(1).astype(str) + "%"
    _delta = _delta.sort_values("Change", ascending=True)

    _delta_fig = go.Figure(go.Bar(
        x=_delta["Change"],
        y=_delta.index,
        orientation="h",
        marker_color=[
            "#e74c3c" if v < 0 else "#2ecc71" if v > 0 else "#95a5a6"
            for v in _delta["Change"]
        ],
        text=[
            f"{'+' if v > 0 else ''}{v} ({p})"
            for v, p in zip(_delta["Change"], _delta["Change %"])
        ],
        textposition="outside",
    ))
    _delta_fig.update_layout(
        title=f"Module Change: {_labels[0]} → {_labels[1]}",
        xaxis_title="Change in Incident Count",
        yaxis_title="",
        height=max(380, len(_delta) * 36),
        margin=dict(l=10, r=60, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(_delta)-0.5,
                     line=dict(color="grey", width=1, dash="dot"))],
    )
    _delta_fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    st.plotly_chart(_delta_fig, use_container_width=True)
    with st.expander("Delta — full table"):
        try:
            st.dataframe(
                _delta.style.background_gradient(subset=["Change"], cmap="RdYlGn"),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(_delta, use_container_width=True)

    # ── Confidence distribution ───────────────────────────────────────────────
    _conf_avail = {_l: _d for _l, _d in dfs.items() if "Confidence" in _d.columns}
    if _conf_avail:
        st.markdown("---")
        st.markdown("### \U0001f3af Confidence Distribution")
        _conf_order = ["High", "Medium", "Low - Needs Review"]
        _conf_rows = []
        for _l, _d in _conf_avail.items():
            for _cat, _cnt in _d["Confidence"].value_counts().items():
                _conf_rows.append({"Period": _l, "Confidence": _cat, "Count": int(_cnt)})
        _conf_long = pd.DataFrame(_conf_rows)
        _conf_cats = [c for c in _conf_order if c in _conf_long["Confidence"].unique()]
        _conf_cats += [c for c in _conf_long["Confidence"].unique() if c not in _conf_order]
        _conf_fig = px.bar(
            _conf_long,
            x="Confidence", y="Count", color="Period",
            barmode="group",
            title="Confidence Level Distribution",
            color_discrete_sequence=_PALETTE,
            category_orders={"Confidence": _conf_cats},
            text="Count",
            height=380,
        )
        _conf_fig.update_traces(textposition="outside")
        _conf_fig.update_layout(
            xaxis_title="", yaxis_title="Incident Count",
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_conf_fig, use_container_width=True)

    # ── Classification method — donut charts per period ───────────────────────
    _meth_avail = {_l: _d for _l, _d in dfs.items() if "Classification Method" in _d.columns}
    if _meth_avail:
        st.markdown("---")
        st.markdown("### \u2699\ufe0f Classification Method Distribution")
        _meth_data = {
            _l: _d["Classification Method"]
                .apply(lambda x: str(x).split(" ")[0])
                .value_counts()
            for _l, _d in _meth_avail.items()
        }
        _n_pie = len(_meth_data)
        _pie_fig = make_subplots(
            rows=1, cols=_n_pie,
            specs=[[{"type": "pie"}] * _n_pie],
            subplot_titles=list(_meth_data.keys()),
        )
        for _pi, (_l, _series) in enumerate(_meth_data.items()):
            _pie_fig.add_trace(
                go.Pie(
                    labels=_series.index.tolist(),
                    values=_series.values.tolist(),
                    name=_l,
                    hole=0.45,
                    textinfo="label+percent",
                    marker_colors=_PALETTE,
                ),
                row=1, col=_pi + 1,
            )
        _pie_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=60, b=10),
            showlegend=True,
        )
        st.plotly_chart(_pie_fig, use_container_width=True)

    # ── Top 15 recurring Module → Type combos ────────────────────────────────
    st.markdown("---")
    st.markdown("### \U0001f501 Top 15 Recurring Issues (Module \u2192 Type)")
    _top_data = {
        _l: (
            _d["Module"].fillna("Unknown").str.strip().str.title()
            + " \u2192 "
            + _d["Type Of Issue"].fillna("Unknown").astype(str).str.strip().str.title()
        ).value_counts()
        for _l, _d in dfs.items()
    }
    _top_df = (
        pd.DataFrame(_top_data)
        .fillna(0).astype(int)
        .sort_values(_labels[0], ascending=False)
    )
    st.plotly_chart(
        _horiz_bar(_top_df, "Issue", "Top 15 Recurring Issues", top_n=15),
        use_container_width=True,
    )
    with st.expander("Full recurring issues table"):
        st.dataframe(_top_df, use_container_width=True)

    # ── New / resolved issues between first two periods ───────────────────────
    st.markdown("---")
    st.markdown(
        f"### \U0001f195 New Issues in **{_labels[1]}** / "
        f"\U0001f7e2 Resolved vs **{_labels[0]}**"
    )
    _combos0 = set(
        (dfs[_labels[0]]["Module"].fillna("") + " \u2192 "
         + dfs[_labels[0]]["Type Of Issue"].fillna("")).unique()
    )
    _combos1 = set(
        (dfs[_labels[1]]["Module"].fillna("") + " \u2192 "
         + dfs[_labels[1]]["Type Of Issue"].fillna("")).unique()
    )
    _new_in_1  = sorted(_combos1 - _combos0)
    _gone_from = sorted(_combos0 - _combos1)
    _nc1, _nc2 = st.columns(2)
    with _nc1:
        st.success(f"**{len(_new_in_1)}** new issue type(s) appeared in {_labels[1]}")
        st.dataframe(
            pd.DataFrame({"New Issues": _new_in_1}),
            use_container_width=True,
            hide_index=True,
            height=min(400, 28 + 24 * min(len(_new_in_1), 15)),
        )
    with _nc2:
        st.info(f"**{len(_gone_from)}** issue type(s) not seen in {_labels[1]}")
        st.dataframe(
            pd.DataFrame({"Resolved Issues": _gone_from}),
            use_container_width=True,
            hide_index=True,
            height=min(400, 28 + 24 * min(len(_gone_from), 15)),
        )

    # --- Download all comparison data as Excel ---
    import io
    from pandas import ExcelWriter
    if st.button("⬇️ Download Comparison Data (Excel)", type="secondary"):
        _buf = io.BytesIO()
        with ExcelWriter(_buf, engine="openpyxl") as _wr:
            _mod_df.to_excel(_wr, sheet_name="Module Distribution")
            _type_df.to_excel(_wr, sheet_name="Type Of Issue Distribution")
            _delta.to_excel(_wr, sheet_name="Module Delta")
            _top_df.to_excel(_wr, sheet_name="Top Recurring Issues")
            pd.DataFrame({"New Issues": _new_in_1}).to_excel(_wr, sheet_name="New Issues", index=False)
            pd.DataFrame({"Resolved Issues": _gone_from}).to_excel(_wr, sheet_name="Resolved Issues", index=False)
            # Add confidence and method if available
            if '_conf_long' in locals():
                _conf_long.to_excel(_wr, sheet_name="Confidence Distribution", index=False)
            if '_meth_data' in locals():
                for _l, _series in _meth_data.items():
                    pd.DataFrame({"Method": _series.index, "Count": _series.values}).to_excel(_wr, sheet_name=f"Method_{_l}", index=False)
        st.download_button(
            label="Download Excel File",
            data=_buf.getvalue(),
            file_name="incident_comparison_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )



def _render_classification_tab(
        sim_thresh, conf_thresh, ai_enabled, ai_threshold, semantic_enabled):
    """Classification Tool tab — full pipeline: upload, classify, review, export."""
    # ── Pre-warm models once per server session ──────────────────────────────────
    # build_classification_models() is @st.cache_resource — first call trains
    # TF-IDF + LogisticRegression + CalibratedClassifier (~1-2 min); all later
    # calls (including inside the button handler) return instantly from cache.
    # By pre-warming here, the Analyze button is never frozen waiting for training.
    if "models_ready" not in st.session_state:
        cache_key = get_cache_key(HISTORICAL_DIR)
        models_on_disk = os.path.exists(
            os.path.join(CACHE_DIR, f"models_{cache_key}.pkl")
        )
        if models_on_disk:
            _init_msg = st.info(
                "⚡ **Loading models from cache** — restoring saved models from disk…  \n"
                "This takes only a few seconds."
            )
            _spinner_msg = "Loading cached models…"
        else:
            _init_msg = st.info(
                "⚙️ **First-time setup** — Training ML models on historical data…  \n"
                "This takes **3–5 minutes** and only happens once. "
                "Models are saved to disk — every restart after this will be instant."
            )
            _spinner_msg = "Building models (one-time setup)…"
        with st.spinner(_spinner_msg):
            build_classification_models()
        st.session_state["models_ready"] = True
        _init_msg.empty()

    # File upload
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            file_bytes = uploaded_file.read()
            try:
                # First pass: get sheet names
                _, _, all_sheets = read_excel_robust(file_bytes)
            except ValueError as ve:
                st.error(str(ve))
                return

            # Sheet selector: auto-pick Dump, otherwise let user choose
            if "Dump" in all_sheets:
                selected_sheet = "Dump"
            else:
                st.warning(
                    f"No sheet named **Dump** found. "
                    f"Available sheets: {all_sheets}. Please select the correct sheet."
                )
                selected_sheet = st.selectbox("Select sheet to classify", all_sheets)

            # Load selected sheet
            df, sheet_name, _ = read_excel_robust(file_bytes, sheet_name=selected_sheet)

            st.subheader("Input Data Preview")
            st.dataframe(df.head(10), width='stretch')
            st.info(f"Total rows: **{len(df)}** | Sheet: **{sheet_name}** | Columns: {len(df.columns)}")

            # Validate required columns
            required_cols = ["Short Description", "Description"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info(f"Available columns: {df.columns.tolist()}")
                return

            # Analyze button
            if st.button("Analyze & Classify Incidents", type="primary"):
                import time as _time
                _t0 = _time.time()
                n_incidents = len(df)
                n_stages = 3 + (1 if ai_enabled else 0) + (1 if semantic_enabled else 0)
                _cur_stage = 1

                # ── Persistent UI elements created once ──────────────────
                master_bar  = st.progress(0.0, text="Starting…")
                stage_status = st.empty()

                # ── Stage 1 — Load models (0 → 12%) ─────────────────────
                stage_status.markdown("**⚙️ Step 1/{} — Confirming models…**".format(n_stages))
                master_bar.progress(0.02, text="Step 1/{} — Loading models…".format(n_stages))
                rules = load_rules()
                similarity_matcher, ml_classifier, ai_reviewer_cached = build_classification_models()
                ai_reviewer = ai_reviewer_cached if ai_enabled else None
                semantic_reviewer = build_semantic_reviewer() if semantic_enabled else None
                if semantic_enabled and semantic_reviewer and not semantic_reviewer._available:
                    st.warning(
                        f"⚠️ **Semantic Review disabled** — model failed to load.  \n"
                        f"Reason: `{semantic_reviewer._init_error}`  \n"
                        "All incidents will show **Semantic Review = Skipped**. "
                        "See sidebar for details."
                    )
                master_bar.progress(0.12, text="Step 1/{} — Models ready ✓".format(n_stages))
                stage_status.markdown(
                    "**✅ Step 1/{n} — Models loaded** &nbsp;|&nbsp; "
                    "{rules} rules &nbsp;|&nbsp; 9,822 training records"
                    .format(n=n_stages, rules=len(rules))
                )

                # ── Stage 2 — Classify incidents (12 → 74%) ──────────────
                _cur_stage = 2
                prog2 = StageProgressBar(
                    master_bar, stage_status, n_incidents,
                    stage_num=_cur_stage, stage_total=n_stages,
                    stage_desc="Classifying incidents",
                    pct_start=12, pct_end=74,
                )
                classification_df = classify_incidents(
                    df, rules, similarity_matcher, ml_classifier, prog2,
                    sim_threshold=sim_thresh, conf_threshold=conf_thresh
                )
                master_bar.progress(0.74, text="Step 2/{} — Classification done ✓".format(n_stages))

                # ── Stage 3 (optional) — AI Review ───────────────────────
                # When both reviewers are enabled, split 74→90% into two slices
                _ai_pct_end = 0.83 if (ai_enabled and semantic_enabled) else 0.90
                if ai_enabled and ai_reviewer:
                    _cur_stage += 1
                    n_review = classification_df["Confidence"].isin(
                        ["Low - Needs Review", "Medium"]
                    ).sum()
                    stage_status.markdown(
                        "**🤖 Step {s}/{n} — AI Review** &nbsp;|&nbsp; "
                        "Reviewing {r} low/medium confidence incidents…"
                        .format(s=_cur_stage, n=n_stages, r=int(n_review))
                    )
                    master_bar.progress(0.76, text="Step {}/{} — AI review…".format(_cur_stage, n_stages))
                    classification_df, ai_promoted_count = apply_ai_review(
                        classification_df, df, ai_reviewer, similarity_matcher,
                        auto_promote_threshold=ai_threshold
                    )
                    master_bar.progress(
                        _ai_pct_end,
                        text="Step {s}/{n} — AI review done ✓ ({p} promoted)".format(
                            s=_cur_stage, n=n_stages, p=ai_promoted_count
                        )
                    )
                else:
                    ai_promoted_count = 0

                # ── Stage N-1 (optional) — Semantic Review ────────────────
                _sem_pct_start = _ai_pct_end if ai_enabled else 0.74
                if semantic_enabled and semantic_reviewer:
                    _cur_stage += 1
                    stage_status.markdown(
                        "**🧠 Step {s}/{n} — Semantic AI Review** &nbsp;|&nbsp; "
                        "Validating all {total:,} incidents with neural language model…"
                        .format(s=_cur_stage, n=n_stages, total=n_incidents)
                    )
                    master_bar.progress(
                        _sem_pct_start + 0.02,
                        text="Step {}/{} — Semantic review…".format(_cur_stage, n_stages)
                    )
                    classification_df, semantic_override_count, semantic_flag_count = \
                        apply_semantic_review(classification_df, df, semantic_reviewer)
                    master_bar.progress(
                        0.92,
                        text="Step {s}/{n} — Semantic review done ✓ "
                             "({o} overridden, {f} flagged)".format(
                                 s=_cur_stage, n=n_stages,
                                 o=semantic_override_count, f=semantic_flag_count
                             )
                    )
                else:
                    semantic_override_count = 0
                    semantic_flag_count = 0

                # ── Final stage — Reports & Export ────────────────────────
                report_stage = n_stages
                stage_status.markdown(
                    "**📊 Step {n}/{n} — Generating summaries & export file…**".format(n=report_stage)
                )
                master_bar.progress(0.92, text="Step {n}/{n} — Building summaries…".format(n=report_stage))
                summary_df = generate_summary(classification_df)
                master_bar.progress(0.95, text="Step {n}/{n} — Building module summary…".format(n=report_stage))
                module_summary_df = generate_module_summary(classification_df)

                # Export
                master_bar.progress(0.98, text="Step {n}/{n} — Exporting Excel…".format(n=report_stage))
                excel_output, result_df = export_excel(
                    df, classification_df, summary_df, module_summary_df
                )
                # Store bytes immediately from this successful export so the
                # download button always has valid data (avoids a second call
                # to export_excel via _regen_excel_bytes which could fail).
                st.session_state["excel_bytes"] = excel_output.getvalue()

                # ── Final: complete banner ────────────────────────────────
                _elapsed = _time.time() - _t0
                master_bar.progress(1.0, text="✅ Complete!")
                stage_status.empty()   # clear stage detail

                # Compute AI-promoted df before storing in session state
                _ai_promo_df = (
                    result_df[result_df["AI Review"] == "Promoted"]
                    if "AI Review" in result_df.columns else pd.DataFrame()
                )

                # Store all results in session state — display happens outside button block
                st.session_state.update({
                    "classification_df":       classification_df.copy(),
                    "result_df":               result_df.copy(),
                    "original_df":             df.copy(),
                    "ai_promoted_df":          _ai_promo_df.copy(),
                    "ai_promoted_count":       ai_promoted_count,
                    "semantic_override_count": semantic_override_count,
                    "semantic_flag_count":     semantic_flag_count,
                    "n_incidents":             n_incidents,
                    "_elapsed":                _elapsed,
                    "semantic_enabled_result": semantic_enabled,
                })
                # Reset inline-correction version counters for fresh analysis
                st.session_state.pop("review_correction_version", None)
                st.session_state.pop("sem_flag_correction_version", None)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        # ── Display classification results (persists across reruns via session state) ──
        if "classification_df" in st.session_state:
            _cdf = st.session_state["classification_df"]
            _rdf = st.session_state["result_df"]
            _ai_promoted_df     = st.session_state.get("ai_promoted_df", pd.DataFrame())
            _ai_promoted_count  = st.session_state.get("ai_promoted_count", 0)
            _sem_override_count = st.session_state.get("semantic_override_count", 0)
            _sem_flag_count     = st.session_state.get("semantic_flag_count", 0)
            _n_incidents        = st.session_state.get("n_incidents", len(_rdf))
            _elapsed_ss         = st.session_state.get("_elapsed", 0.0)
            _sem_enabled        = st.session_state.get("semantic_enabled_result", False)

            # Success banner
            _sem_banner = (
                f" &nbsp;|&nbsp; Semantic overrides: **{_sem_override_count}** "
                f"/ flagged: **{_sem_flag_count}**"
                if _sem_enabled else ""
            )
            st.success(
                f"✅ **Classification complete!** &nbsp; "
                f"Processed **{_n_incidents:,}** incidents in **{_elapsed_ss:.1f}s** "
                f"&nbsp;|&nbsp; AI promoted: **{_ai_promoted_count}**{_sem_banner}"
            )

            # Classification method metrics
            st.subheader("Classification Method Distribution")
            _method_counts = _cdf["Classification Method"].apply(
                lambda x: x.split(" ")[0]
            ).value_counts()
            _c1, _c2, _c3, _c4, _c5, _c6 = st.columns(6)
            with _c1:
                st.metric("Rule-based", _method_counts.get("rule", 0))
            with _c2:
                st.metric("Similarity", _method_counts.get("similarity", 0))
            with _c3:
                st.metric("ML", _method_counts.get("ml", 0))
            with _c4:
                st.metric("Learned", _method_counts.get("learned", 0))
            with _c5:
                st.metric("AI Promoted", _ai_promoted_count)
            with _c6:
                _needs_rev_count = _rdf["Confidence"].isin(["Low - Needs Review", "Medium"]).sum()
                st.metric("Needs Review", int(_needs_rev_count))
            if _sem_enabled:
                _cs1, _cs2 = st.columns(2)
                with _cs1:
                    st.metric("Semantic Overrides", _sem_override_count)
                with _cs2:
                    st.metric("Semantic Flagged", _sem_flag_count)

            # ── Analytics charts ──────────────────────────────────────
            import plotly.graph_objects as _go
            st.markdown("---")
            st.subheader("📊 Classification Analytics")

            _ach_c1, _ach_c2 = st.columns(2)
            with _ach_c1:
                _conf_vc = _cdf["Confidence"].value_counts()
                _conf_color_map = {
                    "High": "#2ecc71", "High (Corrected)": "#27ae60",
                    "Medium": "#f39c12", "Low - Needs Review": "#e74c3c",
                }
                _conf_fig = _go.Figure(_go.Pie(
                    labels=_conf_vc.index.tolist(),
                    values=_conf_vc.values.tolist(),
                    hole=0.48,
                    marker_colors=[_conf_color_map.get(c, "#95a5a6") for c in _conf_vc.index],
                    textinfo="label+percent",
                    textposition="inside",
                    insidetextorientation="radial",
                    hovertemplate="%{label}: %{value} incidents (%{percent})<extra></extra>",
                ))
                _conf_fig.update_layout(
                    title="Confidence Distribution", height=360,
                    margin=dict(l=20, r=120, t=40, b=20),
                    uniformtext=dict(minsize=9, mode="hide"),
                    showlegend=True,
                    legend=dict(orientation="v", x=1.02, y=0.5),
                )
                st.plotly_chart(_conf_fig, use_container_width=True)

            with _ach_c2:
                _meth_vc = _cdf["Classification Method"].apply(
                    lambda x: str(x).split(" ")[0]
                ).value_counts().sort_values()
                _meth_fig = _go.Figure(_go.Bar(
                    x=_meth_vc.values, y=_meth_vc.index, orientation="h",
                    marker_color="#3498db",
                    text=_meth_vc.values, textposition="outside",
                    hovertemplate="%{y}: %{x} incidents<extra></extra>",
                ))
                _meth_fig.update_layout(
                    title="Classification Method Used", height=320,
                    margin=dict(l=0, r=50, t=40, b=0),
                    yaxis_title="", xaxis_title="Incident Count",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _meth_fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
                st.plotly_chart(_meth_fig, use_container_width=True)

            # Module distribution — full width horizontal bar
            _mod_vc = _cdf["Module"].value_counts().sort_values()
            _mod_dist_fig = _go.Figure(_go.Bar(
                x=_mod_vc.values, y=_mod_vc.index, orientation="h",
                marker_color="#9b59b6",
                text=_mod_vc.values, textposition="outside",
                hovertemplate="%{y}: %{x} incidents<extra></extra>",
            ))
            _mod_dist_fig.update_layout(
                title="Module Distribution", height=max(340, len(_mod_vc) * 32),
                margin=dict(l=0, r=60, t=40, b=0),
                yaxis_title="", xaxis_title="Incident Count",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            _mod_dist_fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
            st.plotly_chart(_mod_dist_fig, use_container_width=True)

            # AI-promoted incidents (read-only)
            if _ai_promoted_count > 0 and not _ai_promoted_df.empty:
                with st.expander(
                    f"✅ AI Promoted to High Confidence ({_ai_promoted_count} incidents)",
                    expanded=False
                ):
                    st.success(
                        f"AI reviewed **{_ai_promoted_count}** incidents and upgraded them "
                        "to **High** confidence based on multiple signal agreement."
                    )
                    _ai_promo_cols = [
                        "Number", "Short Description", "Description", "Module",
                        "Type Of Issue", "Classification Method", "AI Confidence", "AI Reason"
                    ]
                    _ai_promo_cols = [c for c in _ai_promo_cols if c in _ai_promoted_df.columns]
                    st.dataframe(
                        _ai_promoted_df[_ai_promo_cols].reset_index(drop=True),
                        use_container_width=True, height=350
                    )

            # ── Needs Review (editable) ──────────────────────────────

            # Build dynamic dropdown options:
            # VALID_* + any values already in the data + any added this session.
            # This gives a full autocomplete-style dropdown while still allowing
            # the user to register brand-new values via the expander below.
            _sess_mods  = st.session_state.get("custom_modules_session", [])
            _sess_types = st.session_state.get("custom_types_session", [])
            _dyn_modules = sorted(
                set(VALID_MODULES)
                | set(_cdf["Module"].dropna().astype(str).tolist())
                | set(_sess_mods),
                key=str.casefold,
            )
            _dyn_types = sorted(
                set(VALID_ISSUE_TYPES)
                | set(_cdf["Type Of Issue"].dropna().astype(str).tolist())
                | set(_sess_types),
                key=str.casefold,
            )

            # ── Register a custom module / issue type ─────────────────
            with st.expander("➕ Register a new Module or Issue Type (adds to dropdown)"):
                st.caption(
                    "Type a new value here and click **Add**. "
                    "It will immediately appear in the Module/Type Of Issue dropdowns "
                    "below and be saved for future sessions."
                )
                _rc1, _rc2, _rc3 = st.columns([3, 3, 1])
                with _rc1:
                    _new_mod_val = st.text_input(
                        "New Module", placeholder="e.g. Mobile App",
                        key="custom_module_input",
                    )
                with _rc2:
                    _new_type_val = st.text_input(
                        "New Issue Type", placeholder="e.g. App Crash",
                        key="custom_type_input",
                    )
                with _rc3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Add", key="add_custom_taxonomy_btn", type="primary"):
                        _added = False
                        if _new_mod_val.strip():
                            register_custom_taxonomy_value("modules", _new_mod_val.strip())
                            _mods_list = st.session_state.setdefault("custom_modules_session", [])
                            if _new_mod_val.strip() not in _mods_list:
                                _mods_list.append(_new_mod_val.strip())
                            _added = True
                        if _new_type_val.strip():
                            register_custom_taxonomy_value("issue_types", _new_type_val.strip())
                            _types_list = st.session_state.setdefault("custom_types_session", [])
                            if _new_type_val.strip() not in _types_list:
                                _types_list.append(_new_type_val.strip())
                            _added = True
                        if _added:
                            st.success("✅ Added — the new value(s) now appear in the dropdowns.")
                            st.rerun()
                        else:
                            st.warning("Enter at least one value before clicking Add.")

            _review_mask = _rdf["Confidence"].isin(["Low - Needs Review", "Medium"])
            _review_df   = _rdf[_review_mask].copy()

            if len(_review_df) > 0:
                st.subheader(f"⚠️ Items Still Needing Review ({len(_review_df)} incidents)")
                st.warning(
                    f"**{len(_review_df)}** incidents remain Low or Medium confidence.  "
                    "**To correct:** edit the Module/Type Of Issue cells.  "
                    "**To confirm already-correct:** tick **Confirm ✓**.  "
                    "Then click **Save**."
                )
                _rev_cols = [
                    "Number", "Short Description", "Description", "Module", "Type Of Issue",
                    "Confidence", "Classification Method", "AI Confidence", "AI Reason"
                ]
                _rev_cols = [c for c in _rev_cols if c in _review_df.columns]
                _rev_slice = _review_df[_rev_cols].reset_index(drop=True)
                _rev_slice.insert(0, "Confirm ✓", False)
                _rev_v     = st.session_state.get("review_correction_version", 0)

                _edited_review = st.data_editor(
                    _rev_slice,
                    column_config={
                        "Confirm ✓": st.column_config.CheckboxColumn(
                            "Confirm ✓",
                            help="Tick to confirm this classification is already correct (no edits needed).",
                            default=False,
                        ),
                        "Module": st.column_config.SelectboxColumn(
                            "Module", options=_dyn_modules,
                            help="Select from the list or register a new value using the ➕ expander above.",
                        ),
                        "Type Of Issue": st.column_config.SelectboxColumn(
                            "Type Of Issue", options=_dyn_types,
                            help="Select from the list or register a new value using the ➕ expander above.",
                        ),
                        **{
                            c: st.column_config.TextColumn(c, disabled=True)
                            for c in _rev_cols if c not in ("Module", "Type Of Issue")
                        },
                    },
                    use_container_width=True,
                    height=min(400, 38 + 35 * len(_rev_slice) + 2),
                    num_rows="fixed",
                    key=f"needs_review_editor_v{_rev_v}",
                    hide_index=True,
                )

                if st.button("💾 Save Corrections from Needs Review", key="save_review_btn"):
                    _n_chg = _apply_inline_corrections(_rev_slice, _edited_review, extra_updates={"Confidence": "High (Corrected)"}, confirm_col="Confirm ✓")
                    if _n_chg > 0:
                        st.session_state["review_correction_version"] = _rev_v + 1
                        st.success(f"✅ {_n_chg} row(s) processed (corrected/confirmed) — all tables updated.")
                        st.rerun()
                    else:
                        st.info("No changes or confirmations detected — edit a cell or tick Confirm ✓ to proceed.")

            elif _ai_promoted_count == 0 and _sem_override_count == 0:
                st.success("All incidents classified with High confidence!")

            # ── Semantic Review results ──────────────────────────────
            if _sem_enabled and "Semantic Review" in _rdf.columns:
                _sem_override_df = _rdf[_rdf["Semantic Review"] == "Overridden"]
                _sem_flag_df     = _rdf[_rdf["Semantic Review"].str.startswith("Flagged", na=False)]

                if len(_sem_override_df) > 0:
                    with st.expander(
                        f"🧠 Semantic AI Overrides ({len(_sem_override_df)} incidents re-classified)",
                        expanded=False
                    ):
                        st.info(
                            f"The semantic model re-classified **{len(_sem_override_df)}** incidents "
                            "where the neural language model strongly disagreed with the original "
                            "module assignment."
                        )
                        _sem_over_cols = [
                            "Number", "Short Description", "Description", "Module",
                            "Type Of Issue", "Classification Method",
                            "Semantic Confidence", "Semantic Reason"
                        ]
                        _sem_over_cols = [c for c in _sem_over_cols if c in _sem_override_df.columns]
                        st.dataframe(
                            _sem_override_df[_sem_over_cols].reset_index(drop=True),
                            use_container_width=True, height=350
                        )

                if len(_sem_flag_df) > 0:
                    with st.expander(
                        f"⚠️ Semantic Flags — Possible Misclassifications ({len(_sem_flag_df)} incidents)",
                        expanded=False
                    ):
                        st.warning(
                            f"The semantic model flagged **{len(_sem_flag_df)}** incidents.  "
                            "**To correct:** edit the Module/Type Of Issue cells.  "
                            "**To confirm already-correct:** tick **Confirm ✓**.  "
                            "Then click **Save**."
                        )
                        _sem_flag_cols = [
                            "Number", "Short Description", "Description", "Module",
                            "Type Of Issue", "Confidence", "Classification Method",
                            "Semantic Review", "Semantic Confidence", "Semantic Reason"
                        ]
                        _sem_flag_cols = [c for c in _sem_flag_cols if c in _sem_flag_df.columns]
                        _sem_flag_slice = _sem_flag_df[_sem_flag_cols].reset_index(drop=True)
                        _sem_flag_slice.insert(0, "Confirm ✓", False)
                        _sem_flag_v     = st.session_state.get("sem_flag_correction_version", 0)

                        _edited_sem_flags = st.data_editor(
                            _sem_flag_slice,
                            column_config={
                                "Confirm ✓": st.column_config.CheckboxColumn(
                                    "Confirm ✓",
                                    help="Tick to confirm this classification is already correct (no edits needed).",
                                    default=False,
                                ),
                                "Module": st.column_config.SelectboxColumn(
                                    "Module", options=_dyn_modules,
                                    help="Select from the list or register a new value using the ➕ expander above.",
                                ),
                                "Type Of Issue": st.column_config.SelectboxColumn(
                                    "Type Of Issue", options=_dyn_types,
                                    help="Select from the list or register a new value using the ➕ expander above.",
                                ),
                                **{
                                    c: st.column_config.TextColumn(c, disabled=True)
                                    for c in _sem_flag_cols if c not in ("Module", "Type Of Issue")
                                },
                            },
                            use_container_width=True,
                            height=350,
                            key=f"sem_flags_editor_v{_sem_flag_v}",
                            hide_index=True,
                        )

                        if st.button(
                            "💾 Save Corrections from Semantic Flags",
                            key="save_sem_flags_btn"
                        ):
                            _n_chg = _apply_inline_corrections(_sem_flag_slice, _edited_sem_flags, extra_updates={"Semantic Review": "Reviewed"}, confirm_col="Confirm ✓")
                            if _n_chg > 0:
                                st.session_state["sem_flag_correction_version"] = _sem_flag_v + 1
                                st.success(f"✅ {_n_chg} row(s) processed (corrected/confirmed) — all tables updated.")
                                st.rerun()
                            else:
                                st.info("No changes or confirmations detected — edit a cell or tick Confirm ✓ to proceed.")

            # ── Full classified data (editable) ───────────────────────
            st.subheader("Classified Data (All)")
            st.info(
                "✏️ **Module** and **Type Of Issue** show a searchable dropdown of all known values.  "
                "To assign a value not yet in the list, use the **➕ Register** expander above — "
                "it instantly adds to the dropdown. Corrections are learned for future runs."
            )
            _disp_cols = [
                "Number", "Short Description", "Description", "Module", "Type Of Issue",
                "Confidence", "Classification Method",
                "AI Review", "AI Confidence", "AI Reason",
                "Semantic Review", "Semantic Confidence", "Semantic Reason"
            ]
            _disp_cols = [c for c in _disp_cols if c in _rdf.columns]
            _all_v     = st.session_state.get("all_correction_version", 0)
            _all_slice = _rdf[_disp_cols].reset_index(drop=True)

            _edited_all = st.data_editor(
                _all_slice,
                column_config={
                    "Module": st.column_config.SelectboxColumn(
                        "Module", options=_dyn_modules,
                        help="Select from the list. To add a new value, use the ➕ Register expander above.",
                    ),
                    "Type Of Issue": st.column_config.SelectboxColumn(
                        "Type Of Issue", options=_dyn_types,
                        help="Select from the list. To add a new value, use the ➕ Register expander above.",
                    ),
                    **{
                        c: st.column_config.TextColumn(c, disabled=True)
                        for c in _disp_cols if c not in ("Module", "Type Of Issue")
                    },
                },
                use_container_width=True,
                height=500,
                key=f"all_data_editor_v{_all_v}",
                hide_index=True,
            )

            if st.button("💾 Save Corrections", key="save_all_btn", type="primary"):
                _n_chg = _apply_inline_corrections(_all_slice, _edited_all)
                if _n_chg > 0:
                    st.session_state["all_correction_version"] = _all_v + 1
                    st.success(
                        f"✅ **{_n_chg}** correction(s) saved — the system has learned "
                        "from your feedback and will apply these patterns to future classifications."
                    )
                    st.rerun()
                else:
                    st.info("No changes detected — edit a Module or Type Of Issue cell to correct a classification.")

            # ── Module → Issue Type interactive drill-down ───────────
            st.markdown("---")
            st.subheader("🔍 Module → Issue Type Breakdown")
            _drill_opts = ["— All Modules —"] + sorted(_cdf["Module"].dropna().unique().tolist())
            _sel_mod_ct = st.selectbox(
                "Select a module to see its issue type breakdown",
                _drill_opts,
                key="cross_module_select",
            )

            if _sel_mod_ct == "— All Modules —":
                # Two-column summary tables side by side
                _sum_df     = generate_summary(_cdf)
                _mod_sum_df = generate_module_summary(_cdf)
                _sc1, _sc2  = st.columns(2)
                with _sc1:
                    st.markdown("**Module Counts**")
                    st.dataframe(_mod_sum_df, use_container_width=True, height=350)
                with _sc2:
                    st.markdown("**Issue Type Counts**")
                    st.dataframe(_sum_df, use_container_width=True, height=350)
                with st.expander("📋 Full Cross-Tabulation (Module × Issue Type)"):
                    st.dataframe(generate_cross_summary(_cdf), use_container_width=True, height=400)
            else:
                _flt = _cdf[_cdf["Module"] == _sel_mod_ct]
                _tc  = _flt["Type Of Issue"].value_counts().sort_values()

                _drill_fig = _go.Figure(_go.Bar(
                    x=_tc.values, y=_tc.index, orientation="h",
                    marker_color="#e67e22",
                    text=_tc.values, textposition="outside",
                    hovertemplate="%{y}: %{x} incidents<extra></extra>",
                ))
                _drill_fig.update_layout(
                    title=f"{_sel_mod_ct} — {len(_flt):,} incidents",
                    height=max(300, len(_tc) * 38),
                    margin=dict(l=0, r=60, t=50, b=0),
                    yaxis_title="", xaxis_title="Incident Count",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _drill_fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
                st.plotly_chart(_drill_fig, use_container_width=True)

                _tc_tbl = pd.DataFrame({
                    "Type Of Issue": _tc.index,
                    "Count": _tc.values,
                    "% of Module": (_tc.values / max(len(_flt), 1) * 100).round(1).astype(str) + "%",
                })
                st.dataframe(
                    _tc_tbl, use_container_width=True, hide_index=True,
                    height=min(420, 44 + 36 * len(_tc_tbl)),
                )

            # ── BigPanda Incidents Section ───────────────────────────────────
            st.markdown("---")
            _bp_df = _cdf[_cdf["Module"] == "BigPanda"].copy()
            if not _bp_df.empty:
                st.subheader("🐼 BigPanda Incidents")

                _bp_count = len(_bp_df)
                _bp_c1, _bp_c2, _bp_c3 = st.columns(3)
                with _bp_c1:
                    st.metric("Total BigPanda Incidents", _bp_count)
                with _bp_c2:
                    st.metric("Unique Services Affected", _bp_df["Type Of Issue"].nunique())
                with _bp_c3:
                    _bp_pct = f"{_bp_count / max(len(_cdf), 1) * 100:.1f}%"
                    st.metric("% of Total Incidents", _bp_pct)

                # Horizontal bar chart — one bar per service/application
                _bp_vc = _bp_df["Type Of Issue"].value_counts().sort_values()
                _bp_fig = _go.Figure(_go.Bar(
                    x=_bp_vc.values,
                    y=_bp_vc.index,
                    orientation="h",
                    marker_color="#e74c3c",
                    text=_bp_vc.values,
                    textposition="outside",
                    hovertemplate="%{y}: %{x} incidents<extra></extra>",
                ))
                _bp_fig.update_layout(
                    title="BigPanda — Service / Application Breakdown",
                    height=max(320, len(_bp_vc) * 42),
                    margin=dict(l=0, r=70, t=50, b=0),
                    yaxis_title="",
                    xaxis_title="Incident Count",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _bp_fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
                st.plotly_chart(_bp_fig, use_container_width=True)

                # Summary table
                st.markdown("**BigPanda Incidents — Service Summary**")
                _bp_summary = _bp_df["Type Of Issue"].value_counts().reset_index()
                _bp_summary.columns = ["Service / Type Of Issue", "Count"]
                _bp_summary["% of BigPanda Incidents"] = (
                    _bp_summary["Count"] / max(_bp_count, 1) * 100
                ).round(1).astype(str) + "%"
                st.dataframe(
                    _bp_summary, use_container_width=True, hide_index=True,
                    height=min(420, 44 + 36 * len(_bp_summary)),
                )

                # Detailed incident table (expandable)
                with st.expander(
                    f"📋 BigPanda Incidents — Full Detail ({_bp_count} incidents)",
                    expanded=False,
                ):
                    _bp_detail_cols = [
                        "Number", "Short Description", "Type Of Issue", "Confidence",
                        "Classification Method",
                    ]
                    _bp_detail_cols = [c for c in _bp_detail_cols if c in _bp_df.columns]
                    st.dataframe(
                        _bp_df[_bp_detail_cols].reset_index(drop=True),
                        use_container_width=True,
                        height=min(500, 44 + 36 * _bp_count),
                    )

            # Download (uses cached excel bytes, refreshed after each correction save)
            st.download_button(
                label="Download Classified Excel",
                data=st.session_state.get("excel_bytes", b""),
                file_name="classified_incidents.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


    # Show auto-promoted rules
    promoted = load_promoted_rules()
    if promoted:
        with st.expander(f"Auto-Promoted Rules ({len(promoted)}) — learned from corrections"):
            for pr in promoted:
                st.markdown(
                    f"**{pr.get('module')} / {pr.get('type_of_issue')}** "
                    f"(from {pr.get('source_corrections', '?')} corrections, "
                    f"priority {pr.get('priority', 97)})"
                )
                st.caption(f"Keywords: {', '.join(pr.get('keywords', []))}")
                st.markdown("---")


def _render_historical_tab():
    """Historical Training Data management tab."""
    # === HISTORICAL DATA MANAGEMENT SECTION ===

    st.markdown("---")
    st.subheader("📂 Manage Historical Training Data")
    st.markdown(
        "Add validated incident data to the **historical database** to improve future "
        "classification accuracy. Files saved here are used by the **Similarity Matcher** "
        "and **ML Classifier** the next time models are rebuilt."
    )

    # ── Column requirements note + sample download ───────────────────────────
    with st.expander("📋 Required Column Format & Sample Template", expanded=True):
        st.markdown(
            """
    **Upload a `.xlsx` file with the following columns:**

    | Column | Status | Description | Example |
    |---|---|---|---|
    | `Short Description` | ✅ Required | Brief one-line summary of the incident | `Unable to confirm delivery in RACPAD` |
    | `Module` | ✅ Required | Business module — must match a value from the **Valid Modules** list | `Agreement` |
    | `Type Of Issue` | ✅ Required | Issue type — must match a value from the **Valid Issue Types** list | `Confirm delivery Issue` |
    | `Description` | ⭐ Recommended | Full incident details / notes text | `Store 01234 cannot confirm delivery...` |
    | `Number` | Optional | Incident number for exact-match correction lookups | `INCTEC1625303` |

    > **Note:** Extra columns (e.g. date, priority, assignee) are preserved but not used during classification.  
    > **Tip:** Including `Description` significantly improves similarity matching accuracy for future incidents.  
    > **Valid Modules:** `{modules}`  
            """.format(modules="`, `".join(VALID_MODULES))
        )

        # Generate and offer the sample template download
        _sample_data = [
            {
                "Number": "INCTEC1000001",
                "Short Description": "Unable to confirm delivery in RACPAD",
                "Description": (
                    "Store is unable to confirm delivery for agreement number 8174605282. "
                    "The system wont let us confirm the delivery the agreement."
                ),
                "Module": "Agreement",
                "Type Of Issue": "Confirm delivery Issue",
            },
            {
                "Number": "INCTEC1000002",
                "Short Description": "Payment not reflecting in system",
                "Description": (
                    "Customer payment was deducted from bank account but not showing in RACPAD. "
                    "Payment pulled but not processed on our end."
                ),
                "Module": "Payment",
                "Type Of Issue": "Data Sync / Integration Failure",
            },
            {
                "Number": "INCTEC1000003",
                "Short Description": "Store showing old location after transfer",
                "Description": (
                    "Employee was transferred to new store but RACPAD still shows the previous "
                    "store assignment. Old store still showing after login."
                ),
                "Module": "Store",
                "Type Of Issue": "Access / Permission Issue",
            },
            {
                "Number": "INCTEC1000004",
                "Short Description": "Agreement void issue — cannot void the agreement",
                "Description": (
                    "Store manager is unable to void the agreement. "
                    "System shows an error when attempting to void."
                ),
                "Module": "Agreement",
                "Type Of Issue": "Agreement Void Issue",
            },
            {
                "Number": "INCTEC1000005",
                "Short Description": "Unable to receive PO in inventory",
                "Description": (
                    "Item arrived at store but cannot be received in the purchase order. "
                    "Option to receive in is not there."
                ),
                "Module": "Inventory",
                "Type Of Issue": "Unable to receive the Purchase order",
            },
        ]
        _sample_df = pd.DataFrame(_sample_data)
        _sample_buf = BytesIO()
        with pd.ExcelWriter(_sample_buf, engine="openpyxl") as _wr:
            _sample_df.to_excel(_wr, index=False, sheet_name="Dump")
        st.download_button(
            label="⬇️ Download Sample Template (.xlsx)",
            data=_sample_buf.getvalue(),
            file_name="historical_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Pre-formatted Excel template with required columns and 5 sample rows.",
        )

    # ── Current historical files summary ─────────────────────────────────────
    os.makedirs(HISTORICAL_DIR, exist_ok=True)
    _existing_files = sorted(
        [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".xlsx")]
    )
    if _existing_files:
        with st.expander(
            f"📁 Current Historical Files ({len(_existing_files)} file(s))", expanded=False
        ):
            _fstat_rows = []
            for _fname in _existing_files:
                _fpath = os.path.join(HISTORICAL_DIR, _fname)
                _fsize_kb = os.path.getsize(_fpath) / 1024
                try:
                    with open(_fpath, "rb") as _fh:
                        _fdata = _fh.read()
                    _fdf, _, _ = read_excel_robust(_fdata)
                    _nrows = len(_fdf)
                    _has_desc = "Yes" if "Description" in _fdf.columns else "No"
                except Exception:
                    _nrows = "?"
                    _has_desc = "?"
                _fstat_rows.append({
                    "File": _fname,
                    "Rows": _nrows,
                    "Has Description": _has_desc,
                    "Size (KB)": f"{_fsize_kb:.1f}",
                })
            st.dataframe(
                pd.DataFrame(_fstat_rows),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No historical files found in `historical/` folder yet.")

    # ── Upload new historical file ────────────────────────────────────────────
    st.markdown("##### ➕ Add New Historical Dump")
    _hist_upload = st.file_uploader(
        "Upload validated historical incident data (.xlsx)",
        type=["xlsx", "xls"],
        key="hist_dump_upload",
        help="File must contain at least: Short Description, Module, Type Of Issue",
    )

    if _hist_upload is not None:
        try:
            _hbytes = _hist_upload.read()
            _hdf, _hsheet, _hsheets = read_excel_robust(_hbytes)

            # Normalize "TypeOfIssue" column name variants
            _col_map = {}
            for _c in _hdf.columns:
                if _c.lower().replace(" ", "").replace("_", "") == "typeofissue":
                    _col_map[_c] = "Type Of Issue"
            if _col_map:
                _hdf = _hdf.rename(columns=_col_map)

            # ── Column validation ─────────────────────────────────────
            _required_cols = ["Short Description", "Module", "Type Of Issue"]
            _missing_cols = [c for c in _required_cols if c not in _hdf.columns]

            if _missing_cols:
                st.error(
                    f"❌ Missing required columns: **{', '.join(_missing_cols)}**  \n"
                    f"Columns found in file: `{', '.join(_hdf.columns.tolist())}`"
                )
            else:
                _n_total = len(_hdf)

                # ── Auto-normalize Module column so case variants don't produce
                # false "unrecognized" warnings. normalize_module() maps
                # "payment" → "Payment", "report" → "Reporting", etc.
                _hdf["Module"] = _hdf["Module"].apply(normalize_module)

                # Case-insensitive matching for Type Of Issue: build a lookup
                # set of all known valid types (lowercased) for comparison.
                _valid_types_lower = {t.strip().lower(): t for t in VALID_ISSUE_TYPES}
                # Normalize Type Of Issue: if a case-insensitive match exists, use canonical form
                def _normalize_type(val):
                    if pd.isna(val):
                        return val
                    stripped = str(val).strip()
                    key = stripped.lower()
                    return _valid_types_lower.get(key, stripped)
                _hdf["Type Of Issue"] = _hdf["Type Of Issue"].apply(_normalize_type)

                _valid_mod_mask  = _hdf["Module"].isin(VALID_MODULES)
                _valid_type_mask = _hdf["Type Of Issue"].isin(VALID_ISSUE_TYPES)
                _n_fully_valid   = (_valid_mod_mask & _valid_type_mask).sum()
                _n_bad_mod       = (~_valid_mod_mask).sum()
                _n_bad_type      = (~_valid_type_mask).sum()
                _has_desc        = "Description" in _hdf.columns

                # Auto-register genuinely new (not just case-variant) values so
                # they appear in dropdowns and are recognized in future sessions.
                if _n_bad_mod > 0:
                    for _nv in _hdf[~_valid_mod_mask]["Module"].dropna().unique():
                        register_custom_taxonomy_value("modules", str(_nv).strip())
                if _n_bad_type > 0:
                    for _nv in _hdf[~_valid_type_mask]["Type Of Issue"].dropna().unique():
                        register_custom_taxonomy_value("issue_types", str(_nv).strip())

                # Summary metrics
                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                _mc1.metric("Total Rows", f"{_n_total:,}")
                _mc2.metric("Fully Valid Rows", f"{_n_fully_valid:,}")
                _mc3.metric("Has Description", "Yes ✅" if _has_desc else "No ⚠️")
                _mc4.metric("Sheet Read", _hsheet)

                if _n_bad_mod > 0:
                    _bad_mods = (
                        _hdf[~_valid_mod_mask]["Module"]
                        .dropna().unique().tolist()
                    )
                    st.info(
                        f"ℹ️ **{_n_bad_mod}** row(s) have novel Module values (auto-registered): "
                        f"`{'`, `'.join(str(x) for x in _bad_mods[:10])}`  \n"
                        "These rows **will be included** in similarity matching and ML training. "
                        "The new values have been added to the known modules list."
                    )
                if _n_bad_type > 0:
                    _bad_types = (
                        _hdf[~_valid_type_mask]["Type Of Issue"]
                        .dropna().unique().tolist()
                    )
                    st.info(
                        f"ℹ️ **{_n_bad_type}** row(s) have novel Type Of Issue values (auto-registered): "
                        f"`{'`, `'.join(str(x) for x in _bad_types[:10])}`  \n"
                        "These rows **will be included** in similarity matching and ML training. "
                        "The new values have been added to the known issue types list."
                    )
                if not _has_desc:
                    st.info(
                        "ℹ️ No `Description` column found. Adding it is recommended for "
                        "better similarity matching accuracy."
                    )

                # Preview
                with st.expander("🔍 Preview uploaded data (first 10 rows)"):
                    _preview_cols = [
                        c for c in [
                            "Number", "Short Description", "Description",
                            "Module", "Type Of Issue",
                        ]
                        if c in _hdf.columns
                    ]
                    st.dataframe(
                        _hdf[_preview_cols].head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Filename input + save
                # Default to a timestamped name so the upload never silently
                # overwrites an existing historical file.
                import datetime as _dt
                _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                _base_fname = _hist_upload.name
                if not _base_fname.endswith(".xlsx"):
                    _base_fname = _base_fname.rsplit(".", 1)[0] + ".xlsx"
                _stem = _base_fname[:-5]  # strip .xlsx
                _default_fname = f"{_stem}_{_ts}.xlsx"

                _save_name = st.text_input(
                    "Save as (filename inside `historical/` folder)",
                    value=_default_fname,
                    help=(
                        "A timestamp is appended by default so existing files are never "
                        "accidentally overwritten. Change the name only if you intentionally "
                        "want to replace an existing file."
                    ),
                )

                _would_overwrite = (
                    _save_name.strip()
                    and os.path.exists(os.path.join(HISTORICAL_DIR, _save_name.strip()))
                )
                _confirm_overwrite = False
                if _would_overwrite:
                    st.error(
                        f"🚫 **`{_save_name.strip()}`** already exists in `historical/`.  \n"
                        "Saving with this name **will replace** the existing file.  \n"
                        "Rename the file above to keep both, or tick the box below to confirm replacement."
                    )
                    _confirm_overwrite = st.checkbox(
                        f"Yes, I want to replace `{_save_name.strip()}` with the new upload",
                        key="hist_overwrite_confirm",
                    )

                if st.button(
                    "💾 Save to Historical Data & Rebuild Models",
                    type="primary",
                    key="save_hist_dump_btn",
                ):
                    _sname = _save_name.strip()
                    if not _sname:
                        st.error("Please enter a filename.")
                    elif not _sname.endswith(".xlsx"):
                        st.error("Filename must end with `.xlsx`.")
                    elif _would_overwrite and not _confirm_overwrite:
                        st.error(
                            "Overwrite not confirmed. Tick the checkbox above or rename the file."
                        )
                    else:
                        try:
                            # Snapshot existing files BEFORE saving (for success message)
                            _files_before = set(
                                f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".xlsx")
                            )

                            # Save the new file — existing files are untouched
                            _save_path = os.path.join(HISTORICAL_DIR, _sname)
                            with open(_save_path, "wb") as _fw:
                                _fw.write(_hbytes)

                            _files_after = set(
                                f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".xlsx")
                            )
                            _kept = sorted(_files_before - {_sname})
                            _action = "replaced" if _sname in _files_before else "added"

                            # Clear Streamlit in-memory caches so next run re-reads the folder
                            load_historical_data.clear()
                            build_classification_models.clear()

                            # Delete stale on-disk .pkl model/embedding caches only
                            # (historical .xlsx files are NOT touched)
                            _n_pkl_deleted = 0
                            if os.path.exists(CACHE_DIR):
                                for _pkl_f in os.listdir(CACHE_DIR):
                                    if _pkl_f.endswith(".pkl"):
                                        try:
                                            os.remove(os.path.join(CACHE_DIR, _pkl_f))
                                            _n_pkl_deleted += 1
                                        except Exception:
                                            pass

                            # Clear models_ready flag so pre-warm banner re-appears
                            st.session_state.pop("models_ready", None)

                            _kept_note = (
                                f"- **{len(_kept)}** existing file(s) preserved: "
                                f"`{'`, `'.join(_kept)}`  \n"
                                if _kept else ""
                            )
                            st.session_state["hist_save_result"] = {
                                "type": "success",
                                "message": (
                                    f"✅ **`{_sname}`** {_action} in `historical/` folder  \n"
                                    f"- **{_n_total:,}** rows {_action} "
                                    f"({_n_fully_valid:,} fully valid)  \n"
                                    f"{_kept_note}"
                                    f"- {_n_pkl_deleted} model cache file(s) cleared  \n"
                                    "- Models will **retrain automatically** on the next "
                                    "classification run."
                                ),
                            }
                        except Exception as _save_err:
                            st.session_state["hist_save_result"] = {
                                "type": "error",
                                "message": f"❌ Failed to save `{_sname}`: {_save_err}",
                            }
                        st.rerun()

                # ── Show save result right below the button ──────────────────
                _save_result = st.session_state.pop("hist_save_result", None)
                if _save_result:
                    if _save_result["type"] == "success":
                        st.success(_save_result["message"])
                    else:
                        st.error(_save_result["message"])

        except Exception as _hex:
            st.error(f"Error reading uploaded file: {_hex}")
            import traceback as _tb
            st.code(_tb.format_exc())


def main():
    st.set_page_config(page_title="RACPad Incident Classification", layout="wide")
    st.title("RACPad Incident Trend Analysis - Classification Tool")
    st.markdown(
        "Upload an Excel file with **Short Description** and **Description** columns. "
        "The tool classifies incidents using a multi-stage hybrid pipeline: "
        "**Rules** → **Historical Similarity** → **ML Prediction** → **AI Review** → **Semantic Review**."
    )

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.markdown("---")

        # Load historical data stats
        hist_df = load_historical_data()
        if not hist_df.empty:
            st.success(f"Historical data: {len(hist_df)} records")
            st.markdown(f"**Modules:** {hist_df['Module'].nunique()}")
            st.markdown(f"**Issue Types:** {hist_df['Type Of Issue'].nunique()}")
        else:
            st.warning("No historical data found in `historical/` folder")

        # Rules info
        rules = load_rules()
        st.markdown(f"**Rules loaded:** {len(rules)}")

        st.markdown("---")
        st.subheader("Thresholds")
        sim_thresh = st.slider("Similarity Threshold", 0.5, 1.0, SIMILARITY_THRESHOLD, 0.05)
        conf_thresh = st.slider("ML Confidence Threshold", 0.2, 0.8, CONFIDENCE_THRESHOLD, 0.05)

        st.markdown("---")
        st.subheader("AI Review")
        ai_enabled = st.checkbox("Enable AI Review", value=True,
                                 help="Run a second-opinion AI pass on Low/Medium confidence incidents")
        ai_threshold = st.slider(
            "AI Promote Threshold", 0.70, 0.95, 0.82, 0.01,
            help="Minimum AI confidence required to auto-upgrade an incident to High"
        )

        st.markdown("---")
        st.subheader("Semantic AI Review")
        semantic_enabled = st.checkbox(
            "Enable Semantic Review", value=True,
            help=(
                "Use a local neural language model (sentence-transformers) to validate ALL "
                "incident classifications — including High confidence ones. "
                "Downloads ~90 MB model on first run. No API key required."
            )
        )

        st.markdown("---")
        st.subheader("Valid Modules")
        for m in VALID_MODULES:
            st.markdown(f"- {m}")

        st.markdown("---")
        st.subheader("Valid Issue Types")
        for t in VALID_ISSUE_TYPES:
            st.markdown(f"- {t}")

    # ── Navigation tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "\U0001f3e0 Classification Tool",
        "\U0001f4ca Trend Comparison",
        "\U0001f4c2 Manage Training Data",
    ])

    with tab2:
        _render_comparison_tab()

    with tab3:
        _render_historical_tab()

    with tab1:
        _render_classification_tab(
            sim_thresh, conf_thresh, ai_enabled, ai_threshold, semantic_enabled
        )

if __name__ == "__main__":
    main()



