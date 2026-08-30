"""
Step 10: Medium→Long routing experiment (CUPID-G pivot).

Why: analysis (07) showed short-vs-long allocation is dominated by fixed_medium.
Pivot: use medium-chain features to decide whether to extend to long.

Routing rule at threshold τ:
  P(helpful_ml | medium_features) > τ  →  run long  (+384 extra tokens vs medium)
  P(helpful_ml | medium_features) ≤ τ  →  keep medium answer  (0 extra tokens)

Baselines:
  fixed_long    always long
  fixed_medium  always medium  (the new short baseline)
  random_τ      random routing at same fraction (100 seeds)
  oracle        route to long iff label_ml=1

Reads:
  {results_dir}/features_medium.jsonl   (from Step 9)

Outputs:
  {output_dir}/ml_routing_results.json
  {output_dir}/ml_routing_results.csv
  {output_dir}/ml_routing_summary.txt
  {output_dir}/ml_routing_pareto.png        (if matplotlib available)
  {output_dir}/ml_predictor.pkl             (final predictor)

Run:
  python 10_medium_long_routing.py
  python 10_medium_long_routing.py --results-dir ../outputs/qwen_main
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from utils.feature_utils import FeatureBuilder

# ---- Token budget constants ----------------------------------------
# Medium chain already paid; delta = long - medium
MEDIUM_TOKENS = 384
LONG_TOKENS   = 768
DELTA_TOKENS  = LONG_TOKENS - MEDIUM_TOKENS   # 384 extra to go medium→long

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 3)
DOMAIN_COLS = {"g1_is_gsm8k", "g1_is_math", "g1_math_level"}
SCRIPT_DIR = Path(__file__).resolve().parent


def load_features(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} medium-feature records")
    return records


def resolve_path(path: str) -> Path:
    """Resolve CLI paths relative to analysis script dir, not caller cwd."""
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def validate_no_leakage_features(X: pd.DataFrame) -> None:
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
            + ". These leak the medium→long utility label."
        )


def build_classifier(cfg: Dict):
    try:
        import lightgbm as lgb
        lgb_cfg = cfg["classifiers"]["lightgbm"]
        return lgb.LGBMClassifier(
            n_estimators=lgb_cfg["n_estimators"],
            num_leaves=lgb_cfg["num_leaves"],
            learning_rate=lgb_cfg["learning_rate"],
            class_weight=lgb_cfg["class_weight"],
            verbose=-1, n_jobs=1,
            random_state=cfg["classifiers"]["random_state"],
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        rf_cfg = cfg["classifiers"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            class_weight=rf_cfg["class_weight"],
            n_jobs=1,
            random_state=rf_cfg["random_state"],
        )


def get_oof_probs(X, y, clf, n_folds, random_state) -> np.ndarray:
    oof = np.zeros(len(y))
    cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        model = clone(clf)
        model.fit(X.iloc[tr].values, y.iloc[tr].values)
        if hasattr(model, "predict_proba"):
            oof[val] = model.predict_proba(X.iloc[val].values)[:, 1]
        else:
            oof[val] = model.decision_function(X.iloc[val].values)
    return oof


def simulate(oof_probs, threshold, medium_correct, long_correct) -> Dict:
    routed  = oof_probs > threshold
    correct = np.where(routed, long_correct, medium_correct)
    extra   = np.where(routed, DELTA_TOKENS, 0)
    return {
        "threshold":           float(threshold),
        "accuracy":            float(correct.mean()),
        "avg_delta_tokens":    float(extra.mean()),
        "avg_total_tokens":    float(MEDIUM_TOKENS + extra.mean()),
        "fraction_routed":     float(routed.mean()),
        "token_savings_pct":   float(100.0 * (1.0 - extra.mean() / DELTA_TOKENS)),
    }


def random_baseline(fraction, medium_correct, long_correct, n_seeds=100) -> Dict:
    n   = len(medium_correct)
    rng = np.random.default_rng(0)
    accs = []
    for _ in range(n_seeds):
        mask = rng.random(n) < fraction
        accs.append(np.where(mask, long_correct, medium_correct).mean())
    extra_mean = fraction * DELTA_TOKENS
    return {
        "accuracy":          float(np.mean(accs)),
        "accuracy_std":      float(np.std(accs)),
        "avg_delta_tokens":  float(extra_mean),
        "avg_total_tokens":  float(MEDIUM_TOKENS + extra_mean),
        "fraction_routed":   float(fraction),
        "token_savings_pct": float(100.0 * (1.0 - extra_mean / DELTA_TOKENS)),
    }


def compute_baselines(medium_correct, long_correct, utility_labels) -> Dict:
    oracle_mask    = utility_labels.astype(bool)
    oracle_correct = np.where(oracle_mask, long_correct, medium_correct)
    oracle_extra   = np.where(oracle_mask, DELTA_TOKENS, 0)
    return {
        "fixed_long": {
            "accuracy":          float(long_correct.mean()),
            "avg_delta_tokens":  float(DELTA_TOKENS),
            "avg_total_tokens":  float(LONG_TOKENS),
            "fraction_routed":   1.0,
            "token_savings_pct": 0.0,
        },
        "fixed_medium": {
            "accuracy":          float(medium_correct.mean()),
            "avg_delta_tokens":  0.0,
            "avg_total_tokens":  float(MEDIUM_TOKENS),
            "fraction_routed":   0.0,
            "token_savings_pct": 100.0,
        },
        "oracle": {
            "accuracy":          float(oracle_correct.mean()),
            "avg_delta_tokens":  float(oracle_extra.mean()),
            "avg_total_tokens":  float(MEDIUM_TOKENS + oracle_extra.mean()),
            "fraction_routed":   float(oracle_mask.mean()),
            "token_savings_pct": float(100.0 * (1.0 - oracle_extra.mean() / DELTA_TOKENS)),
        },
    }


def dominates(row, fixed_medium_acc) -> bool:
    """True if this CUPID-G point is dominated by fixed_medium (same tokens, worse accuracy)."""
    return row["accuracy"] < fixed_medium_acc and row["fraction_routed"] < 0.01


def build_summary(baselines, cupidg_rows, cv_auc) -> str:
    lines = [
        "CUPID-G analysis — Medium→Long Routing Results",
        "=" * 52, "",
        f"Predictor AUC (5-fold CV): {cv_auc:.4f}", "",
        "Baselines:",
    ]
    for k, v in baselines.items():
        lines.append(
            f"  {k:14s}: acc={v['accuracy']:.4f}  "
            f"avg_total_tok={v['avg_total_tokens']:.0f}  "
            f"savings={v['token_savings_pct']:.1f}%"
        )
    lines.append("")

    med_acc  = baselines["fixed_medium"]["accuracy"]
    long_acc = baselines["fixed_long"]["accuracy"]
    acc_gap  = long_acc - med_acc

    # Count dominated rows
    dom = sum(1 for r in cupidg_rows if r["accuracy"] <= med_acc and r["fraction_routed"] > 0)
    lines.append(f"CUPID-G threshold sweep: {len(cupidg_rows)} points, "
                 f"{dom} dominated by fixed_medium")
    lines.append("")

    # Best efficiency: max accuracy gain per token spent
    best_eff, best_score = None, -1e9
    for row in cupidg_rows:
        if row["fraction_routed"] < 1e-6:
            continue
        gain = row["accuracy"] - med_acc
        cost_frac = row["avg_delta_tokens"] / DELTA_TOKENS
        score = gain / max(cost_frac, 1e-6)
        if score > best_score:
            best_score, best_eff = score, row

    if best_eff and best_eff["accuracy"] > med_acc:
        gap_recovered = 100 * (best_eff["accuracy"] - med_acc) / max(acc_gap, 1e-6)
        lines.append("Best efficiency point (CUPID-G):")
        lines.append(f"  τ = {best_eff['threshold']:.2f}")
        lines.append(f"  Accuracy = {best_eff['accuracy']:.4f}  "
                     f"({gap_recovered:.1f}% of medium→long gap recovered)")
        lines.append(f"  Token savings vs fixed_long = {best_eff['token_savings_pct']:.1f}%")
        lines.append(f"  vs random routing: "
                     f"{best_eff['accuracy'] - best_eff.get('random_accuracy', float('nan')):.4f} gain")
    else:
        lines.append("WARNING: CUPID-G cannot outperform fixed_medium at any threshold.")
        lines.append("  → Signal too weak for medium→long routing on this dataset.")
        lines.append("  → Consider G2+G3-only features (better cross-domain transfer).")

    lines.append("")
    lines.append(f"Oracle upper bound: acc={baselines['oracle']['accuracy']:.4f}  "
                 f"savings={baselines['oracle']['token_savings_pct']:.1f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",  default="../outputs/qwen_main")
    parser.add_argument("--pipeline-config",  default="../pipeline/config.yaml")
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--feature-set",    choices=["G4", "G4_nodomain", "G2G3"], default="G4_nodomain",
                        help="G4=all features, G4_nodomain=drop domain cols, G2G3=PRM+consistency only")
    args = parser.parse_args()

    results_root = resolve_path(args.results_dir)
    out_dir    = resolve_path(args.output_dir) if args.output_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = results_root / "features_medium.jsonl"
    if not feat_path.exists():
        logger.error(f"Missing: {feat_path}. Run Step 9 first.")
        sys.exit(1)

    cfg_path = resolve_path(args.pipeline_config)
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"classifiers": {
            "random_forest": {"n_estimators": 200, "max_depth": 8,
                              "class_weight": "balanced", "random_state": 42},
            "lightgbm": {"n_estimators": 200, "num_leaves": 31,
                         "learning_rate": 0.05, "class_weight": "balanced"},
            "random_state": 42, "cv_folds": 5,
        }}

    records = load_features(feat_path)
    X_g4, y = FeatureBuilder.records_to_dataframe(records, "G4")

    # Select feature set
    if args.feature_set == "G4_nodomain":
        drop = [c for c in DOMAIN_COLS if c in X_g4.columns]
        X = X_g4.drop(columns=drop)
        logger.info(f"Feature set: G4-no-domain ({len(X.columns)} features, dropped {drop})")
    elif args.feature_set == "G2G3":
        X_g2, _ = FeatureBuilder.records_to_dataframe(records, "G2")
        X_g3, _ = FeatureBuilder.records_to_dataframe(records, "G3")
        X = pd.concat([X_g2, X_g3], axis=1)
        logger.info(f"Feature set: G2+G3 ({len(X.columns)} features)")
    else:
        X = X_g4
        logger.info(f"Feature set: G4 full ({len(X.columns)} features)")
    validate_no_leakage_features(X)

    medium_correct = np.array([int(r["medium_orm_correct"]) for r in records])
    long_correct   = np.array([int(r["long_orm_correct"])   for r in records])
    utility_labels = y.values

    n_pos   = int(y.sum())
    n_total = len(y)
    logger.info(f"helpful_ml=1: {n_pos}/{n_total} ({100*n_pos/n_total:.1f}%)")
    if n_pos < 20:
        logger.error("Too few positives for reliable evaluation.")
        sys.exit(1)

    clf        = build_classifier(cfg)
    n_folds    = cfg["classifiers"]["cv_folds"]
    rs         = cfg["classifiers"]["random_state"]

    # CV AUC
    min_class = int(y.value_counts().min())
    folds = min(n_folds, min_class)
    cv_obj = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
    auc_scores = cross_val_score(clf, X.values, y.values, cv=cv_obj, scoring="roc_auc", n_jobs=1)
    cv_auc_mean = float(auc_scores.mean())
    logger.info(f"\nCV AUC ({args.feature_set}): {cv_auc_mean:.4f} ± {auc_scores.std():.4f}")

    # OOF probs for simulation
    logger.info("Computing OOF probabilities...")
    oof_probs = get_oof_probs(X, y, clf, folds, rs)

    # Baselines
    baselines = compute_baselines(medium_correct, long_correct, utility_labels)
    logger.info("\nBaselines:")
    for k, v in baselines.items():
        logger.info(f"  {k:14s}: acc={v['accuracy']:.4f}  "
                    f"tokens={v['avg_total_tokens']:.0f}  "
                    f"savings={v['token_savings_pct']:.1f}%")

    # Threshold sweep
    logger.info(f"\nThreshold sweep ({args.feature_set}):")
    cupidg_rows = []
    med_acc = baselines["fixed_medium"]["accuracy"]
    for tau in THRESHOLDS:
        row  = simulate(oof_probs, tau, medium_correct, long_correct)
        rand = random_baseline(row["fraction_routed"], medium_correct, long_correct)
        row["random_accuracy"]        = rand["accuracy"]
        row["random_accuracy_std"]    = rand["accuracy_std"]
        row["gain_over_random"]       = row["accuracy"] - rand["accuracy"]
        row["beats_fixed_medium"]     = row["accuracy"] > med_acc
        cupidg_rows.append(row)
        marker = "✓" if row["beats_fixed_medium"] else " "
        logger.info(
            f"  {marker} τ={tau:.2f}  acc={row['accuracy']:.4f}  "
            f"rand={rand['accuracy']:.4f}  "
            f"gain={row['gain_over_random']:+.4f}  "
            f"savings={row['token_savings_pct']:.1f}%  "
            f"routed={row['fraction_routed']:.2f}"
        )

    n_beats_medium = sum(1 for r in cupidg_rows if r["beats_fixed_medium"])
    logger.info(f"\nCUPID-G beats fixed_medium at {n_beats_medium}/{len(cupidg_rows)} thresholds")

    # Save
    output = {
        "feature_set": args.feature_set,
        "cv_auc":      cv_auc_mean,
        "baselines":   baselines,
        "cupidg":      cupidg_rows,
        "meta": {
            "n_total":        n_total,
            "n_helpful_ml":   n_pos,
            "helpful_ml_rate": float(n_pos / n_total),
            "medium_tokens":  MEDIUM_TOKENS,
            "long_tokens":    LONG_TOKENS,
            "delta_tokens":   DELTA_TOKENS,
        },
    }
    json_path = out_dir / f"ml_routing_results_{args.feature_set.lower()}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    rows_out = []
    for k, v in baselines.items():
        rows_out.append({"strategy": k, "threshold": "-", **v})
    for row in cupidg_rows:
        rows_out.append({"strategy": f"cupidg_{args.feature_set}", **row})
    csv_path = out_dir / f"ml_routing_results_{args.feature_set.lower()}.csv"
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)

    summary = build_summary(baselines, cupidg_rows, cv_auc_mean)
    txt_path = out_dir / f"ml_routing_summary_{args.feature_set.lower()}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"\nSaved: {json_path}\n       {csv_path}\n       {txt_path}")
    logger.info("\n" + summary)

    # Save predictor
    final_clf = clone(clf)
    final_clf.fit(X.values, y.values)
    pkl_path = out_dir / f"ml_predictor_{args.feature_set.lower()}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"clf": final_clf, "feature_names": list(X.columns),
                     "feature_set": args.feature_set}, f)
    logger.info(f"Saved predictor: {pkl_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = [r["avg_total_tokens"] for r in cupidg_rows]
        ys = [r["accuracy"]         for r in cupidg_rows]
        ax.plot(xs, ys, "b-o", markersize=4, label=f"CUPID-G ({args.feature_set})")
        colors = {"fixed_long": "r", "fixed_medium": "g", "oracle": "purple"}
        for k, v in baselines.items():
            ax.axhline(v["accuracy"], color=colors[k], linestyle="--", alpha=0.7, label=k)
            ax.scatter([v["avg_total_tokens"]], [v["accuracy"]], color=colors[k], zorder=5, s=60)
        ax.set_xlabel("Avg total tokens per problem (medium + delta)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"CUPID-G Medium→Long Pareto ({args.feature_set})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig_path = out_dir / f"ml_routing_pareto_{args.feature_set.lower()}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot: {fig_path}")
    except Exception as e:
        logger.info(f"Plot skipped: {e}")

    logger.info("\nStep 10 complete.")
    logger.info("\nTo run all 3 feature sets:")
    logger.info("  python 10_medium_long_routing.py --feature-set G4")
    logger.info("  python 10_medium_long_routing.py --feature-set G4_nodomain")
    logger.info("  python 10_medium_long_routing.py --feature-set G2G3")


if __name__ == "__main__":
    main()
