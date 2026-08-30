"""
Step 5: Train LR / LightGBM / Random Forest on G1-G4 feature groups.
        Report AUC via stratified k-fold cross-validation.

This is the primary Go/No-Go decision step for CUPID-G. Inspect robustness
diagnostics before treating the random-CV G4 AUC as strong evidence.

AUC thresholds (from config.yaml go_nogo section):
  < 0.60  -> ABORT CUPID-G, start Plan B immediately
  0.60-0.65 -> WARNING: add more features, hard deadline 1 week
  0.65-0.70 -> conditional: verify with second model
  >= 0.70  -> FULL CUPID-G

Outputs:
  {output_dir}/auc_results.json    (machine-readable)
  {output_dir}/auc_results.csv     (human-readable table)
  {output_dir}/auc_results.txt     (verdict text)
  Single-group runs use auc_results_G1/G2/G3/G4.* instead.
  {output_dir}/feature_importance_G4_{classifier}.csv  (feature importance)

Run:
  python 05_train_evaluate.py --config config.yaml
  python 05_train_evaluate.py --config config.yaml --group G4  # single group
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from utils.feature_utils import FeatureBuilder


# ------------------------------------------------------------------
# Classifiers
# ------------------------------------------------------------------

def build_classifiers(cfg: dict) -> Dict:
    """Build classifier instances from config."""
    clfs = {}
    clf_cfg = cfg["classifiers"]

    if clf_cfg["logistic_regression"]["enabled"]:
        lr_cfg = clf_cfg["logistic_regression"]
        clfs["LR"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=lr_cfg["C"],
                max_iter=lr_cfg["max_iter"],
                class_weight=lr_cfg["class_weight"],
                random_state=cfg["classifiers"]["random_state"],
            )),
        ])

    if clf_cfg["lightgbm"]["enabled"]:
        try:
            import lightgbm as lgb
            lgb_cfg = clf_cfg["lightgbm"]
            clfs["LGB"] = lgb.LGBMClassifier(
                n_estimators=lgb_cfg["n_estimators"],
                num_leaves=lgb_cfg["num_leaves"],
                learning_rate=lgb_cfg["learning_rate"],
                class_weight=lgb_cfg["class_weight"],
                verbose=lgb_cfg["verbose"],
                n_jobs=lgb_cfg.get("n_jobs", 1),
                random_state=cfg["classifiers"]["random_state"],
            )
        except ImportError:
            logger.warning("LightGBM not installed. Skipping LGB.")

    if clf_cfg["random_forest"]["enabled"]:
        rf_cfg = clf_cfg["random_forest"]
        clfs["RF"] = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            class_weight=rf_cfg["class_weight"],
            n_jobs=rf_cfg.get("n_jobs", 1),
            random_state=rf_cfg["random_state"],
        )

    return clfs


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_group(
    X: pd.DataFrame,
    y: pd.Series,
    classifiers: Dict,
    n_folds: int,
    random_state: int,
) -> Dict[str, float]:
    """Run CV for all classifiers on one feature group. Returns {clf_name: mean_auc}."""
    if y.nunique() < 2:
        raise ValueError("AUC requires both positive and negative labels.")
    min_class = int(y.value_counts().min())
    if min_class < 2:
        raise ValueError("At least two examples per class are required for cross-validation.")
    n_folds = min(n_folds, min_class)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {}

    for clf_name, clf in classifiers.items():
        scores = cross_val_score(
            clf, X.values, y.values,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        results[clf_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": [float(s) for s in scores],
        }
        logger.info(f"    {clf_name:4s}: AUC = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return results


def validate_no_leakage_features(X: pd.DataFrame) -> None:
    """Fail fast if ground-truth-derived columns accidentally enter training."""
    banned_fragments = [
        "orm_correct",
        "long_correct",
        "medium_correct",
        "utility",
        "label",
    ]
    leaking = [
        col for col in X.columns
        if any(fragment in col.lower() for fragment in banned_fragments)
    ]
    if leaking:
        raise ValueError(
            "Ground-truth-derived feature(s) detected: "
            + ", ".join(leaking)
            + ". These leak the utility label and must not be used for AUC."
        )


def evaluate_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classifiers: Dict,
) -> Dict[str, float]:
    """Train on one split and report holdout AUC for all classifiers."""
    results = {}
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return results

    for clf_name, clf in classifiers.items():
        model = clone(clf)
        model.fit(X_train.values, y_train.values)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test.values)[:, 1]
        else:
            scores = model.decision_function(X_test.values)
        results[clf_name] = float(roc_auc_score(y_test.values, scores))
    return results


def evaluate_dataset_breakdowns(
    X: pd.DataFrame,
    y: pd.Series,
    records: List[Dict],
    classifiers: Dict,
    n_folds: int,
    random_state: int,
) -> Dict[str, Dict]:
    """
    Additional non-decision diagnostics:
      - within-dataset CV
      - train GSM8K -> test MATH
      - train MATH -> test GSM8K
    """
    datasets = pd.Series([r["dataset"] for r in records], index=X.index, name="dataset")
    diagnostics = {"within_dataset_cv": {}, "cross_dataset": {}}

    for dataset_name in sorted(datasets.unique()):
        mask = datasets == dataset_name
        X_sub = X.loc[mask]
        y_sub = y.loc[mask]
        min_class = int(y_sub.value_counts().min()) if y_sub.nunique() == 2 else 0
        if min_class < 2:
            continue
        folds = min(n_folds, min_class)
        diagnostics["within_dataset_cv"][dataset_name] = evaluate_group(
            X_sub, y_sub, classifiers, folds, random_state
        )

    if set(datasets.unique()) >= {"gsm8k", "math"}:
        gsm_mask = datasets == "gsm8k"
        math_mask = datasets == "math"
        # Strip domain-identity columns before cross-dataset holdout.
        # Keeping g1_is_gsm8k / g1_is_math / g1_math_level would let the
        # classifier trivially detect the domain shift rather than measuring
        # true utility-prediction generalization.
        domain_cols = {"g1_is_gsm8k", "g1_is_math", "g1_math_level"}
        X_cross = X.drop(columns=[c for c in domain_cols if c in X.columns])
        diagnostics["cross_dataset"]["train_gsm8k_test_math"] = evaluate_holdout(
            X_cross.loc[gsm_mask], y.loc[gsm_mask],
            X_cross.loc[math_mask], y.loc[math_mask],
            classifiers,
        )
        diagnostics["cross_dataset"]["train_math_test_gsm8k"] = evaluate_holdout(
            X_cross.loc[math_mask], y.loc[math_mask],
            X_cross.loc[gsm_mask], y.loc[gsm_mask],
            classifiers,
        )

    return diagnostics


def get_verdict(auc: float, thresholds: dict) -> Tuple[str, str]:
    """Return (emoji, verdict text) based on AUC thresholds."""
    abort = thresholds["auc_abort"]
    warn_low = thresholds["auc_warn_low"]
    warn_high = thresholds["auc_warn_high"]
    go = thresholds["auc_conditional"]

    if auc < abort:
        return "🔴", f"ABORT CUPID-G. AUC={auc:.4f} < {abort}. Start Plan B-1 TODAY."
    elif auc < warn_high:
        return "🟡", f"WARNING. AUC={auc:.4f} in [{warn_low},{warn_high}). Add features; 1-week hard deadline."
    elif auc < go:
        return "🟡", f"CONDITIONAL. AUC={auc:.4f} in [{warn_high},{go}). Verify with second model before full run."
    else:
        return "🟢", f"GO. AUC={auc:.4f} >= {go}. Proceed with full CUPID-G."


def get_feature_importance(clf, feature_names: List[str], clf_name: str) -> pd.DataFrame:
    """Extract feature importances where available."""
    if clf_name == "LGB":
        importances = clf.feature_importances_
    elif clf_name == "RF":
        importances = clf.feature_importances_
    elif clf_name == "LR":
        importances = np.abs(clf.named_steps["clf"].coef_[0])
    else:
        return pd.DataFrame()
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--group", choices=["G1", "G2", "G3", "G4", "all"], default="all")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_dir"])
    features_path = out_dir / "features.jsonl"

    if not features_path.exists():
        logger.error(f"Missing: {features_path}. Run 04_build_features.py first.")
        sys.exit(1)

    # Load feature records
    records = []
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} feature records")
    if not records:
        logger.error("No feature records found.")
        sys.exit(1)

    n_pos = sum(r["label"] for r in records)
    n_neg = len(records) - n_pos
    logger.info(f"Label distribution: {n_pos} positive ({100*n_pos/len(records):.1f}%), "
                f"{n_neg} negative")

    if min(n_pos, n_neg) < 20:
        logger.error("Too few examples in at least one class (<20). Results will be unreliable. "
                     "Increase dataset size or check label computation.")
        sys.exit(1)

    classifiers = build_classifiers(cfg)
    if not classifiers:
        logger.error("No classifiers enabled or available. Check config and dependencies.")
        sys.exit(1)
    cv_folds = cfg["classifiers"]["cv_folds"]
    random_state = cfg["classifiers"]["random_state"]
    thresholds = cfg["go_nogo"]

    groups_to_run = ["G1", "G2", "G3", "G4"] if args.group == "all" else [args.group]

    all_results = {}
    diagnostics = {}

    logger.info("\n" + "=" * 60)
    logger.info("CUPID-G Phase 1: AUC Evaluation")
    logger.info("=" * 60)

    for group_name in groups_to_run:
        logger.info(f"\n--- Feature Group: {group_name} ---")
        X, y = FeatureBuilder.records_to_dataframe(records, group_name)
        validate_no_leakage_features(X)
        logger.info(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

        group_results = evaluate_group(X, y, classifiers, cv_folds, random_state)
        all_results[group_name] = group_results

        diagnostics[group_name] = evaluate_dataset_breakdowns(
            X, y, records, classifiers, cv_folds, random_state
        )

    # ---- Summary table ----
    rows = []
    for group_name, group_res in all_results.items():
        for clf_name, res in group_res.items():
            rows.append({
                "Group": group_name,
                "Classifier": clf_name,
                "AUC_mean": res["mean"],
                "AUC_std": res["std"],
            })

    df_summary = pd.DataFrame(rows)
    pivot = df_summary.pivot(index="Group", columns="Classifier", values="AUC_mean")

    logger.info("\n" + "=" * 60)
    logger.info("AUC SUMMARY TABLE")
    logger.info("=" * 60)
    logger.info("\n" + pivot.to_string())

    if "G4" in diagnostics:
        logger.info("\n" + "=" * 60)
        logger.info("G4 ROBUSTNESS DIAGNOSTICS")
        logger.info("=" * 60)
        g4_diag = diagnostics["G4"]
        for dataset_name, dataset_results in g4_diag.get("within_dataset_cv", {}).items():
            logger.info(f"\nWithin-dataset CV: {dataset_name}")
            for clf_name, res in dataset_results.items():
                logger.info(f"    {clf_name:4s}: AUC = {res['mean']:.4f} ± {res['std']:.4f}")
        for split_name, split_results in g4_diag.get("cross_dataset", {}).items():
            if not split_results:
                logger.info(f"\nCross-dataset: {split_name}: skipped (single-class split)")
                continue
            logger.info(f"\nCross-dataset: {split_name}")
            for clf_name, auc in split_results.items():
                logger.info(f"    {clf_name:4s}: AUC = {auc:.4f}")

    # ---- Go/No-Go decision ----
    logger.info("\n" + "=" * 60)
    logger.info("GO / NO-GO DECISION")
    logger.info("=" * 60)

    if "G4" in all_results:
        g4_results = all_results["G4"]
        best_auc = max(r["mean"] for r in g4_results.values())
        best_clf = max(g4_results.items(), key=lambda x: x[1]["mean"])[0]
        emoji, verdict = get_verdict(best_auc, thresholds)
        logger.info(f"\nBest G4 AUC: {best_auc:.4f} (model: {best_clf})")
        logger.info(f"Decision: {emoji} {verdict}")
    else:
        best_auc = None
        best_clf = None
        emoji = "❓"
        verdict = "G4 not evaluated. Run with --group all for final decision."
        logger.warning(verdict)

    # ---- Feature importance on G4 ----
    if "G4" in all_results and classifiers:
        X_g4, y_g4 = FeatureBuilder.records_to_dataframe(records, "G4")
        importance_clf = "LGB" if "LGB" in classifiers else ("RF" if "RF" in classifiers else "LR")
        # Refit on full data for importance
        clf_g4 = classifiers[importance_clf]
        clf_g4.fit(X_g4.values, y_g4.values)
        fi_df = get_feature_importance(clf_g4, list(X_g4.columns), importance_clf)
        if not fi_df.empty:
            fi_path = out_dir / f"feature_importance_G4_{importance_clf.lower()}.csv"
            fi_df.to_csv(fi_path, index=False)
            logger.info(f"\nTop 10 features (G4, {importance_clf}):")
            logger.info(fi_df.head(10).to_string(index=False))
            logger.info(f"Saved: {fi_path}")

    # ---- Save outputs ----
    # JSON
    output_stem = "auc_results" if args.group == "all" else f"auc_results_{args.group}"
    json_out = out_dir / f"{output_stem}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "diagnostics": diagnostics,
            "best_g4_auc": best_auc,
            "best_g4_clf": best_clf if best_auc is not None else None,
            "verdict": verdict,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved: {json_out}")

    # CSV
    csv_out = out_dir / f"{output_stem}.csv"
    df_summary.to_csv(csv_out, index=False)
    logger.info(f"Saved: {csv_out}")

    # Text verdict
    txt_out = out_dir / f"{output_stem}.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("CUPID-G Phase 1 — AUC Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(pivot.to_string() + "\n\n")
        if best_auc is None:
            f.write("Best G4 AUC: N/A\n")
        else:
            f.write(f"Best G4 AUC: {best_auc:.4f} ({best_clf})\n")
        f.write(f"Decision: {emoji} {verdict}\n")
        if "G4" in diagnostics:
            f.write("\nG4 robustness diagnostics:\n")
            g4_diag = diagnostics["G4"]
            for dataset_name, dataset_results in g4_diag.get("within_dataset_cv", {}).items():
                f.write(f"  Within {dataset_name} CV:\n")
                for clf_name, res in dataset_results.items():
                    f.write(f"    {clf_name}: {res['mean']:.4f} ± {res['std']:.4f}\n")
            for split_name, split_results in g4_diag.get("cross_dataset", {}).items():
                f.write(f"  {split_name}:\n")
                if not split_results:
                    f.write("    skipped (single-class split)\n")
                for clf_name, auc in split_results.items():
                    f.write(f"    {clf_name}: {auc:.4f}\n")
        f.write("\nThresholds:\n")
        for k, v in thresholds.items():
            f.write(f"  {k}: {v}\n")
    logger.info(f"Saved: {txt_out}")

    logger.info("\nStep 5 complete.")
    logger.info(f"\nFINAL VERDICT: {emoji} {verdict}")


if __name__ == "__main__":
    main()
