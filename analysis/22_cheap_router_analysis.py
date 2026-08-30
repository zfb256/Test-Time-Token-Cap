"""
Step 22: Explain and stress-test cheap router.

Cheap router is surprisingly strong. This script checks:
  - feature importance for cheap/format routers
  - correlations with labels and correctness
  - top-K selected examples by dataset / math level / medium correctness
  - manual-readable error slices
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.feature_utils import FeatureBuilder

DOMAIN_COLS = {"g1_is_gsm8k", "g1_is_math", "g1_math_level"}
MEDIUM_TOKENS = 384
LONG_TOKENS = 768
DELTA_TOKENS = LONG_TOKENS - MEDIUM_TOKENS


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_map(path: Path) -> Dict[str, Dict]:
    return {r["problem_id"]: r for r in load_jsonl(path)}


def medium_output_features(records: List[Dict], medium_map: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    ids = []
    for r in records:
        pid = r["problem_id"]
        med = medium_map[pid]
        chain = med["chains"][0]["text"] if med.get("chains") else ""
        token_count = med["chains"][0].get("token_count", 0) if med.get("chains") else 0
        rows.append({
            "m_token_count": float(token_count),
            "m_char_len": float(len(chain)),
            "m_line_count": float(chain.count("\n") + 1 if chain else 0),
            "m_has_boxed": float("\\boxed" in chain),
            "m_has_therefore": float("therefore" in chain.lower()),
            "m_has_final": float("final" in chain.lower()),
            "m_digit_count": float(sum(ch.isdigit() for ch in chain)),
            "m_digit_density": float(sum(ch.isdigit() for ch in chain) / max(len(chain), 1)),
        })
        ids.append(pid)
    return pd.DataFrame(rows, index=ids).fillna(0.0)


def build_feature_sets(records: List[Dict], medium_map: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    X_g1, _ = FeatureBuilder.records_to_dataframe(records, "G1")
    X_g3, _ = FeatureBuilder.records_to_dataframe(records, "G3")
    X_out = medium_output_features(records, medium_map)
    X_g1_nodomain = X_g1.drop(columns=[c for c in DOMAIN_COLS if c in X_g1.columns])
    sets = {
        "format": pd.concat([X_g3, X_out], axis=1),
        "cheap": pd.concat([X_g1, X_g3, X_out], axis=1),
        "cheap_nodomain": pd.concat([X_g1_nodomain, X_g3, X_out], axis=1),
    }
    return {k: v.loc[:, v.nunique(dropna=False) > 1] for k, v in sets.items()}


def oof_probs(X: pd.DataFrame, y: pd.Series, clf, folds: int, random_state: int):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    probs = np.zeros(len(y))
    fold_models = []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(clf)
        model.fit(X.iloc[train_idx].values, y.iloc[train_idx].values)
        probs[val_idx] = model.predict_proba(X.iloc[val_idx].values)[:, 1]
        fold_models.append(model)
    return probs, fold_models


def build_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )


def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def feature_importance_table(X, y, out_dir: Path, prefix: str):
    rf = build_rf()
    rf.fit(X.values, y.values)
    rf_imp = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": rf.feature_importances_,
    }).sort_values("rf_importance", ascending=False)
    rf_imp.to_csv(out_dir / f"{prefix}_rf_importance.csv", index=False)

    perm = permutation_importance(
        rf,
        X.values,
        y.values,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    perm_df = pd.DataFrame({
        "feature": X.columns,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    perm_df.to_csv(out_dir / f"{prefix}_permutation_importance.csv", index=False)
    return rf_imp, perm_df


def cv_permutation_importance_table(X, y, out_dir: Path, prefix: str, folds: int):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        rf = build_rf()
        rf.fit(X.iloc[train_idx].values, y.iloc[train_idx].values)
        perm = permutation_importance(
            rf,
            X.iloc[val_idx].values,
            y.iloc[val_idx].values,
            scoring="roc_auc",
            n_repeats=10,
            random_state=42 + fold,
            n_jobs=1,
        )
        for feature, mean, std in zip(X.columns, perm.importances_mean, perm.importances_std):
            rows.append({
                "fold": fold,
                "feature": feature,
                "perm_importance_mean": mean,
                "perm_importance_std": std,
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / f"{prefix}_cv_permutation_importance_detail.csv", index=False)
    summary = (
        detail.groupby("feature")
        .agg(
            cv_perm_importance_mean=("perm_importance_mean", "mean"),
            cv_perm_importance_std=("perm_importance_mean", "std"),
        )
        .reset_index()
        .sort_values("cv_perm_importance_mean", ascending=False)
    )
    summary.to_csv(out_dir / f"{prefix}_cv_permutation_importance.csv", index=False)
    return summary


def selected_slice_table(meta: pd.DataFrame, probs: np.ndarray, fractions: List[float]) -> pd.DataFrame:
    rows = []
    n = len(meta)
    order = np.argsort(-probs)
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        selected = np.zeros(n, dtype=bool)
        selected[order[:k]] = True
        sub = meta[selected]
        rows.append({
            "budget_fraction": frac,
            "n_selected": k,
            "selected_helpful_rate": sub["helpful_ml"].mean(),
            "overall_helpful_rate": meta["helpful_ml"].mean(),
            "selected_medium_acc": sub["medium_correct"].mean(),
            "selected_long_acc": sub["long_correct"].mean(),
            "selected_math_rate": (sub["dataset"] == "math").mean(),
            "selected_gsm8k_rate": (sub["dataset"] == "gsm8k").mean(),
            "selected_mean_math_level": sub["math_level"].mean(),
            "selected_mean_medium_tokens": sub["medium_tokens"].mean(),
            "selected_boxed_rate": sub["m_has_boxed"].mean(),
            "selected_extraction_rate": sub["g3_extraction_rate"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../outputs/qwen_main")
    args = parser.parse_args()

    out_dir = resolve_path(args.results_dir)
    records = load_jsonl(out_dir / "features_medium.jsonl")
    medium_map = load_jsonl_map(out_dir / "inference_medium.jsonl")
    _, y = FeatureBuilder.records_to_dataframe(records, "G4")
    y = y.reset_index(drop=True)
    feature_sets = build_feature_sets(records, medium_map)
    X_g3, _ = FeatureBuilder.records_to_dataframe(records, "G3")

    X_out = medium_output_features(records, medium_map).reset_index(drop=True)
    meta = pd.DataFrame({
        "problem_id": [r["problem_id"] for r in records],
        "dataset": [r["dataset"] for r in records],
        "math_level": [r.get("math_level", 0) for r in records],
        "helpful_ml": y.values,
        "medium_correct": [int(r["medium_orm_correct"]) for r in records],
        "long_correct": [int(r["long_orm_correct"]) for r in records],
        "medium_tokens": X_out["m_token_count"].values,
        "m_has_boxed": X_out["m_has_boxed"].values,
        "g3_extraction_rate": X_g3.reset_index(drop=True)["g3_extraction_rate"].values,
    })

    folds = min(5, int(y.value_counts().min()))
    outputs = []
    for fs_name, X in feature_sets.items():
        X = X.reset_index(drop=True)
        probs, _ = oof_probs(X, y, build_rf(), folds, 42)
        auc = roc_auc_score(y, probs)
        logger.info("%s RF OOF AUC %.4f", fs_name, auc)
        rf_imp, perm_imp = feature_importance_table(X, y, out_dir, f"cheap_router_{fs_name}")
        cv_perm_imp = cv_permutation_importance_table(X, y, out_dir, f"cheap_router_{fs_name}", folds)
        slices = selected_slice_table(meta, probs, [0.1, 0.2, 0.3, 0.4])
        slices.insert(0, "feature_set", fs_name)
        slices.insert(1, "auc_oof", auc)
        outputs.append(slices)

        # Save top false positives/false negatives for manual inspection.
        inspect = meta.copy()
        inspect["pred_helpful"] = probs
        inspect["rank"] = inspect["pred_helpful"].rank(ascending=False, method="first")
        inspect["error_type"] = np.where(
            (inspect["pred_helpful"] >= np.quantile(probs, 0.8)) & (inspect["helpful_ml"] == 0),
            "high_score_not_helpful",
            np.where(
                (inspect["pred_helpful"] <= np.quantile(probs, 0.2)) & (inspect["helpful_ml"] == 1),
                "low_score_helpful",
                "",
            ),
        )
        inspect[inspect["error_type"] != ""].sort_values(
            ["error_type", "pred_helpful"],
            ascending=[True, False],
        ).head(80).to_csv(out_dir / f"cheap_router_{fs_name}_errors.csv", index=False)

    slice_df = pd.concat(outputs, ignore_index=True)
    slice_path = out_dir / "cheap_router_selected_slices.csv"
    slice_df.to_csv(slice_path, index=False)

    # Simple feature-label correlations for cheap_nodomain.
    X_corr = feature_sets["cheap_nodomain"].reset_index(drop=True)
    corr_rows = []
    for col in X_corr.columns:
        if X_corr[col].std() == 0:
            continue
        corr_rows.append({
            "feature": col,
            "pearson_with_helpful": float(np.corrcoef(X_corr[col].values, y.values)[0, 1]),
            "mean_if_helpful": float(X_corr.loc[y.values == 1, col].mean()),
            "mean_if_not_helpful": float(X_corr.loc[y.values == 0, col].mean()),
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_with_helpful", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(out_dir / "cheap_router_feature_correlations.csv", index=False)
    logger.info("Saved: %s", slice_path)
    logger.info("Saved: %s", out_dir / "cheap_router_feature_correlations.csv")


if __name__ == "__main__":
    main()
