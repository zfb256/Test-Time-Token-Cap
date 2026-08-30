"""
Step 32: Second-model sanity evaluation.

For a small second-model run, evaluate:
  Q1: medium->long accuracy gap
  Q2: helpful_ml rate
  Q3: cheap completion/router signals

This script intentionally avoids PRM features so it can run for non-Qwen models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pipeline"))
from utils.math_verify import MathVerifier


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SCRIPT_DIR / p


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_map(path: Path) -> Dict[str, Dict]:
    out = {}
    for record in load_jsonl(path):
        pid = record["problem_id"]
        if pid in out:
            raise ValueError(f"{path}: duplicate problem_id={pid}")
        out[pid] = record
    return out


def first_chain(record: Dict) -> Dict:
    chains = record.get("chains") or []
    return chains[0] if chains else {"text": "", "token_count": 0}


def question_features(record: Dict) -> Dict[str, float]:
    q = record.get("question", "")
    return {
        "q_len_words": float(len(q.split())),
        "q_len_chars": float(len(q)),
        "q_digit_count": float(sum(ch.isdigit() for ch in q)),
        "q_math_level": float(record.get("math_level", 0)),
        "q_is_math": float(record.get("dataset") == "math"),
    }


def medium_features(record: Dict, medium: Dict, medium_budget: int) -> Dict[str, float]:
    chain = first_chain(medium)
    text = chain.get("text", "")
    tokens = int(chain.get("token_count", 0))
    finish_reason = chain.get("finish_reason")
    return {
        "m_token_count": float(tokens),
        "m_hit_max": float(
            finish_reason == "length"
            if finish_reason is not None
            else tokens == medium_budget
        ),
        "m_char_len": float(len(text)),
        "m_line_count": float(text.count("\n") + 1 if text else 0),
        "m_has_boxed": float("\\boxed" in text),
        "m_has_final": float("final" in text.lower()),
        "m_has_therefore": float("therefore" in text.lower()),
        "m_digit_count": float(sum(ch.isdigit() for ch in text)),
        "m_digit_density": float(sum(ch.isdigit() for ch in text) / max(len(text), 1)),
    }


def eval_mask(mask, medium_correct, long_correct, medium_budget, long_budget):
    mask = np.array(mask, dtype=bool)
    correct = np.where(mask, long_correct, medium_correct)
    avg_budget = medium_budget + mask.mean() * (long_budget - medium_budget)
    return {
        "routed": float(mask.mean()),
        "accuracy": float(correct.mean()),
        "avg_budget_tokens": float(avg_budget),
        "savings_vs_long_pct": float(100.0 * (1.0 - avg_budget / long_budget)),
    }


def random_accuracy(frac, medium_correct, long_correct, n_seeds=500):
    rng = np.random.default_rng(0)
    n = len(medium_correct)
    if not n:
        return float("nan")
    k = min(n, max(0, int(round(frac * n))))
    if k == 0:
        return float(np.mean(medium_correct))
    if k == n:
        return float(np.mean(long_correct))
    accs = []
    for _ in range(n_seeds):
        idx = rng.choice(n, size=k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        accs.append(np.where(mask, long_correct, medium_correct).mean())
    return float(np.mean(accs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../outputs/llama_validation")
    parser.add_argument("--medium-name", default="medium")
    parser.add_argument("--long-name", default="long")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    medium_map = load_jsonl_map(out_dir / f"inference_{args.medium_name}.jsonl")
    long_map = load_jsonl_map(out_dir / f"inference_{args.long_name}.jsonl")
    if set(medium_map) != set(long_map):
        raise SystemExit("medium and long problem_id sets differ")
    ids = sorted(medium_map)
    if not ids:
        raise SystemExit("medium and long runs are empty")
    verifier = MathVerifier()

    rows = []
    feature_rows = []
    for pid in ids:
        m = medium_map[pid]
        l = long_map[pid]
        for field in ("dataset", "question", "answer", "math_level"):
            default = 0 if field == "math_level" else None
            if m.get(field, default) != l.get(field, default):
                raise SystemExit(
                    f"medium/long {field} mismatch for problem_id={pid}"
                )
        if len(m.get("chains", [])) != 1 or len(l.get("chains", [])) != 1:
            raise SystemExit(
                f"expected one chain per run for problem_id={pid}"
            )
        m_chain = first_chain(m)
        l_chain = first_chain(l)
        m_text = m_chain.get("text", "")
        l_text = l_chain.get("text", "")
        medium_budget = int(m.get("budget", {}).get("max_new_tokens", 384))
        long_budget = int(l.get("budget", {}).get("max_new_tokens", 768))
        medium_correct = verifier.verify(m_text, m["answer"], m["dataset"])
        if l_text == m_text:
            long_correct = medium_correct
        else:
            long_correct = verifier.verify(l_text, l["answer"], l["dataset"])
        row = {
            "problem_id": pid,
            "dataset": m["dataset"],
            "math_level": m.get("math_level", 0),
            "medium_correct": medium_correct,
            "long_correct": long_correct,
            "helpful_ml": (not medium_correct) and long_correct,
            "medium_tokens": int(m_chain.get("token_count", 0)),
            "long_tokens": int(l_chain.get("token_count", 0)),
            "medium_budget": medium_budget,
            "long_budget": long_budget,
            "medium_hit_max": (
                m_chain.get("finish_reason") == "length"
                if m_chain.get("finish_reason") is not None
                else int(m_chain.get("token_count", 0)) == medium_budget
            ),
            "medium_has_boxed": "\\boxed" in m_chain.get("text", ""),
        }
        rows.append(row)
        feats = {}
        feats.update(question_features(m))
        feats.update(medium_features(m, m, medium_budget))
        feature_rows.append(feats)

    df = pd.DataFrame(rows)
    if (
        df["medium_budget"].nunique() != 1
        or df["long_budget"].nunique() != 1
        or int(df["medium_budget"].iloc[0]) >= int(df["long_budget"].iloc[0])
    ):
        raise SystemExit(
            "medium/long budget metadata is inconsistent or not increasing"
        )
    X = pd.DataFrame(feature_rows).fillna(0.0)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    y = df["helpful_ml"].astype(int)

    summary_rows = []
    for dataset, sub in [("all", df), *df.groupby("dataset")]:
        summary_rows.append({
            "dataset": dataset,
            "n": len(sub),
            "medium_acc": float(sub["medium_correct"].mean()),
            "long_acc": float(sub["long_correct"].mean()),
            "helpful_ml_rate": float(sub["helpful_ml"].mean()),
            "long_harm_rate": float((sub["medium_correct"] & ~sub["long_correct"]).mean()),
            "medium_hit_max_rate": float(sub["medium_hit_max"].mean()),
            "medium_boxed_rate": float(sub["medium_has_boxed"].mean()),
        })

    route_rows = []
    medium_correct = df["medium_correct"].values.astype(bool)
    long_correct = df["long_correct"].values.astype(bool)
    medium_budget = int(df["medium_budget"].iloc[0])
    long_budget = int(df["long_budget"].iloc[0])
    for name, mask in {
        "rule_hit_max": df["medium_hit_max"].values,
        "rule_no_boxed": ~df["medium_has_boxed"].values,
    }.items():
        row = eval_mask(mask, medium_correct, long_correct, medium_budget, long_budget)
        rand = random_accuracy(row["routed"], medium_correct, long_correct)
        row.update({"strategy": name, "random_accuracy": rand, "gain_over_random": row["accuracy"] - rand})
        route_rows.append(row)

    if y.nunique() >= 2 and min(y.value_counts()) >= 2:
        folds = min(5, int(y.value_counts().min()))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
        models = {
            "LR": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
            ]),
            "RF": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=args.seed,
                n_jobs=1,
            ),
        }
        for model_name, model in models.items():
            aucs = cross_val_score(model, X.values, y.values, cv=cv, scoring="roc_auc", n_jobs=1)
            probs = cross_val_predict(model, X.values, y.values, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            for frac in [0.2, 0.3, 0.4]:
                k = max(1, int(round(frac * len(probs))))
                mask = np.zeros(len(probs), dtype=bool)
                mask[np.argsort(-probs)[:k]] = True
                row = eval_mask(mask, medium_correct, long_correct, medium_budget, long_budget)
                rand = random_accuracy(frac, medium_correct, long_correct)
                row.update({
                    "strategy": f"cheap_{model_name}_top{int(frac * 100)}",
                    "auc_cv_mean": float(aucs.mean()),
                    "auc_oof": float(roc_auc_score(y.values, probs)),
                    "random_accuracy": rand,
                    "gain_over_random": row["accuracy"] - rand,
                })
                route_rows.append(row)

    detail_path = out_dir / "second_model_sanity_detail.csv"
    summary_path = out_dir / "second_model_sanity_summary.csv"
    route_path = out_dir / "second_model_sanity_routing.csv"
    txt_path = out_dir / "second_model_sanity_summary.txt"
    df.to_csv(detail_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(route_rows).to_csv(route_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Second Model Sanity Summary\n")
        f.write("===========================\n\n")
        f.write(pd.DataFrame(summary_rows).to_string(index=False))
        f.write("\n\nRouting:\n")
        f.write(pd.DataFrame(route_rows).to_string(index=False))
        f.write("\n")

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {route_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
