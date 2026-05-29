"""
Pre-build all classification models and save to disk cache.

Run this script once after any change to historical data or model code:

    python prebuild_models.py

After this runs, the Streamlit app starts in seconds for every user —
no training wait, no first-user penalty.

How it works:
  1. Mocks the Streamlit library so app.py can be imported without a server
  2. Calls the same training logic used by the live app
  3. Saves models to .cache/models_<hash>.pkl  (loaded by app.py on startup)
     and      .cache/embeddings_<hash>.pkl  (similarity search index)
  The <hash> is derived from historical file contents, so the app auto-detects
  when historical data changes and will prompt a rebuild.

When to re-run:
  - After adding / replacing files in historical/
  - After submitting corrections that meaningfully change the training set
  - After upgrading Python / scikit-learn (pickle format may change)
"""

import sys
import os
import time

# ---------------------------------------------------------------------------
# Minimal Streamlit stub — makes @st.cache_data / @st.cache_resource behave
# as pass-through decorators, and silences all other st.* calls.
# Must be installed BEFORE any app.py import.
# ---------------------------------------------------------------------------
class _StreamlitStub:
    """Pass-through stub for Streamlit decorators; no-ops for everything else."""

    def cache_data(self, func=None, **kwargs):
        if callable(func):
            return func          # @st.cache_data  (no-arg form)
        return lambda f: f       # @st.cache_data(show_spinner=False, ...)

    def cache_resource(self, func=None, **kwargs):
        if callable(func):
            return func
        return lambda f: f

    def __getattr__(self, _):
        # Silently absorb st.info(), st.warning(), st.spinner(), etc.
        return lambda *a, **kw: _NullCtx()


class _NullCtx:
    """Context-manager no-op (handles `with st.spinner(): ...`)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def __call__(self, *a, **kw): return _NullCtx()


sys.modules["streamlit"] = _StreamlitStub()

# ---------------------------------------------------------------------------
# Always run from the directory where this script lives, so relative paths
# (historical/, .cache/) resolve correctly regardless of where it's invoked.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from app import (  # noqa: E402 (import after sys.path manipulation)
    get_cache_key,
    load_historical_data,
    SimilarityMatcher,
    MLClassifier,
    AIReviewer,
    load_models_cache,
    save_models_cache,
    HISTORICAL_DIR,
    CACHE_DIR,
)

# ---------------------------------------------------------------------------
# Pickle compatibility fix:
# app.py saves ML objects whose class paths are bound to "__main__" (because
# Streamlit executes app.py as __main__).  When we load those objects here,
# pickle looks for __main__.MLClassifier — which lives in *this* script, not
# in app.py.  Injecting the classes into our __main__ makes both directions
# work: "Streamlit saved + prebuild loads" and "prebuild saved + Streamlit loads".
# ---------------------------------------------------------------------------
_this_main = sys.modules["__main__"]
for _cls_name in ("MLClassifier", "AIReviewer", "SimilarityMatcher"):
    setattr(_this_main, _cls_name, locals()[_cls_name])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _cache_file(cache_key: str) -> str:
    return os.path.join(CACHE_DIR, f"models_{cache_key}.pkl")


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------
def main():
    print()
    print("=" * 62)
    print("  RACPad Incident Classification — Model Pre-Builder")
    print("=" * 62)

    total_start = time.time()

    # Step 1 — Historical data
    print("\n[1/4] Loading historical data …")
    t = time.time()
    hist_df = load_historical_data()
    if hist_df.empty:
        print(f"\n  ERROR: No historical data found in '{HISTORICAL_DIR}/'")
        print("  Place .xlsx training files there and retry.")
        sys.exit(1)
    print(f"       {len(hist_df):,} records loaded  ({_fmt(time.time()-t)})")

    # Cache key — derived from historical file metadata
    cache_key = get_cache_key(HISTORICAL_DIR)
    print(f"\n[2/4] Cache key : {cache_key}")
    print(f"       Cache file: {_cache_file(cache_key)}")

    # Step 2 — Check for existing cache
    existing_ml, existing_ai = load_models_cache(cache_key)
    if existing_ml is not None and existing_ai is not None:
        print("\n  Models already cached — nothing to do.")
        print("  To force a full rebuild, delete the file above and re-run.")
        print()
        return

    # Step 3 — Similarity / embedding index
    print("\n[3/4] Building TF-IDF similarity index …")
    t = time.time()
    SimilarityMatcher(hist_df)   # internally saves embeddings to disk
    print(f"       Similarity index ready  ({_fmt(time.time()-t)})")

    # Step 4 — ML + AI models
    print("\n[4/4] Training ML + AI models …")
    print("       (LogisticRegression + CalibratedSVC — may take 3-5 min)")

    t = time.time()
    print("       Training MLClassifier …", end="", flush=True)
    ml_classifier = MLClassifier(hist_df)
    ml_time = time.time() - t
    print(f"  done  ({_fmt(ml_time)})")

    t = time.time()
    print("       Training AIReviewer   …", end="", flush=True)
    ai_reviewer = AIReviewer(hist_df)
    ai_time = time.time() - t
    print(f"  done  ({_fmt(ai_time)})")

    # Save
    print(f"\n       Saving to {_cache_file(cache_key)} …", end="", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    save_models_cache(cache_key, ml_classifier, ai_reviewer)
    print("  saved")

    # Summary
    total = time.time() - total_start
    print()
    print("=" * 62)
    print(f"  Done in {_fmt(total)}")
    print()
    print("  The Streamlit app will now start instantly for all users.")
    print("  Re-run this script whenever historical data changes.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pre-build RACPad classification models and save to disk cache."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing cached models and rebuild from scratch."
    )
    args = parser.parse_args()

    if args.force:
        key = get_cache_key(HISTORICAL_DIR)
        old = _cache_file(key)
        if os.path.exists(old):
            os.remove(old)
            print(f"Deleted {old} — will rebuild from scratch.")

    main()
