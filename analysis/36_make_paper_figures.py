"""Regenerate paper figures from canonical CSVs without running inference."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ROUTING_OUTPUTS = ROOT / "outputs" / "three_arm_routing"
THREE_ARM_OUTPUTS = ROOT / "outputs" / "three_arm_expanded"
FIGURES = ROOT / "paper" / "figures"
MODELS = (("Qwen", "qwen_seed42"), ("Llama", "llama_seed42"))


def harm_row(run_dir, contrast):
    rows = pd.read_csv(THREE_ARM_OUTPUTS / run_dir / "three_arm_harm_summary.csv")
    return rows[
        (rows["contrast"] == contrast)
        & (rows["dataset"] == "all")
        & (rows["definition"] == "complete_plurality_correct")
    ].iloc[0]


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_three_arm_routing(scope="core_3seed"):
    """Routing curves inside the three-arm design, where routing a record is an
    exactly observed swap of its arm-B aggregate for its arm-C aggregate."""
    curve = pd.read_csv(ROUTING_OUTPUTS / "routing_curve.csv")
    summary = pd.read_csv(ROUTING_OUTPUTS / "routing_summary.csv")
    series = (
        ("cap_hit_rank", "Cap-hit rank", "#1f77b4", "o"),
        ("learned_logistic", "Cheap learned router", "#d62728", "s"),
        ("random", "Random routing", "#7f7f7f", None),
    )

    panels = [("Qwen", "qwen", scope), ("Llama", "llama", scope)]
    if not summary[(summary["model"] == "r1")
                   & (summary["scope"] == "r1_core_3seed")].empty:
        panels.append(("R1", "r1", "r1_core_3seed"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.0),
                             sharey=False)
    for ax, (label, model, model_scope) in zip(axes, panels):
        rows = curve[(curve["model"] == model) & (curve["scope"] == model_scope)]
        facts = summary[(summary["model"] == model)
                        & (summary["scope"] == model_scope)].iloc[0]
        base = 100 * facts["base_accuracy"]
        extended = 100 * facts["extended_accuracy"]
        cap_hit = 100 * facts["cap_hit_fraction"]

        first = ax is axes[0]
        ax.axhline(extended, color="#2ca02c", linestyle="--", linewidth=1.0,
                   label="Fixed long (arm C)" if first else None)
        ax.axhline(base, color="#8c564b", linestyle=":", linewidth=1.0,
                   label="Fixed base (arm B)" if first else None)
        ax.axvline(cap_hit, color="#bbbbbb", linewidth=0.8, zorder=0)
        for method, name, color, marker in series:
            points = rows[rows["method"] == method].sort_values("route_fraction")
            ax.plot(100 * points["route_fraction"], 100 * points["accuracy"],
                    label=name, color=color, marker=marker, markersize=3.2,
                    linewidth=1.3)
        ax.annotate(f"route every cap hit ({cap_hit:.1f}%)",
                    xy=(cap_hit + 3, base + 0.45 * (extended - base)),
                    fontsize=6.8, color="#555555")
        ax.set_title(f"{label}: arm B {base:.1f} to arm C {extended:.1f}", fontsize=8.5)
        ax.set_xlabel("Records routed (%)", fontsize=8)
        ax.set_ylabel("Plurality accuracy (%)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper center",
                   bbox_to_anchor=(0.5 * len(panels), -0.26), ncol=3,
                   fontsize=7.2, frameon=False, handlelength=1.8,
                   columnspacing=1.2)
    save(fig, "three_arm_routing")
    plt.close(fig)


def repair_row(run_dir):
    rows = pd.read_csv(THREE_ARM_OUTPUTS / run_dir / "three_arm_summary.csv")
    return rows[(rows["dataset"] == "all")
                & (rows["definition"] == "strict_all_wrong")].iloc[0]


def make_answer_churn():
    """Both directions at once: resampling moves answers, continuation does not.

    Left panel counts corrections of a complete-but-wrong aggregate; right
    panel counts reversals of an aggregate that was already correct. Each is
    a rate over the arm-A-eligible population for that direction, so the two
    panels have different denominators and are not compared across panels.
    """
    resample_color, continue_color = "#4c78a8", "#e45756"
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=False)

    # Both panels contrast the same two quantities: what resampling moves, and
    # what continuation adds on top of it. The continuation bar is therefore
    # the incremental B->C count in both, never the cumulative A->C count.
    panels = (
        ("Corrections", "wrong $\\rightarrow$ right",
         [(repair_row(run)["n_eligible"],
           repair_row(run)["samecap_repair_count"],
           repair_row(run)["incremental_helpful_count"]) for _, run in MODELS]),
        ("Reversals", "right $\\rightarrow$ wrong",
         [(harm_row(run, "A_to_B_resampling")["n_eligible"],
           harm_row(run, "A_to_B_resampling")["harm_count"],
           harm_row(run, "B_to_C_tokens")["harm_count"]) for _, run in MODELS]),
    )

    width = 0.34
    for ax, (title, subtitle, data) in zip(axes, panels):
        x = range(len(MODELS))
        top = max(100 * d[1] / d[0] for d in data)
        for offset, pick, label, color in (
            (-width / 2, 1, "Resampling (A$\\rightarrow$B)", resample_color),
            (width / 2, 2, "Continuation adds (B$\\rightarrow$C)",
             continue_color),
        ):
            bars = ax.bar([i + offset for i in x],
                          [100 * d[pick] / d[0] for d in data],
                          width, label=label, color=color)
            for bar, d in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02 * top,
                        f"{int(d[pick])}/{int(d[0]):,}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(x), [name for name, _ in MODELS], fontsize=9)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_ylabel("Rate (%)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, top * 1.30)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.18), ncol=2,
                   fontsize=8, frameon=False)
    save(fig, "answer_churn")
    plt.close(fig)


def make_three_arm_protocol():
    """Figure 1: the three-arm protocol drawn on one shared token axis.

    Layout follows a 900x540 grid with the y axis inverted, so positions read
    top-down. Font sizes carry a single scale factor because matplotlib sizes
    text in points while the layout is in data units.
    """
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, FancyArrowPatch

    BLUE, GREEN, ORANGE = "#4C72B0", "#55A868", "#DD8452"
    INK, MUTED, LINE = "#22252A", "#6B7280", "#9CA3AF"
    PANEL, BORDER = "#FFFFFF", "#E3E6EA"
    FS = 0.74                      # points per layout unit

    fig, ax = plt.subplots(figsize=(10.6, 6.05))
    ax.set_xlim(0, 900)
    ax.set_ylim(514, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def card(x, y, w, h, r=12, ec=BORDER, fc=PANEL, lw=1.0, z=3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))

    def bar(x, y, w, h, colour, r=2.5, z=4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    linewidth=0, facecolor=colour, zorder=z))

    def txt(x, y, s, size, color=INK, ha="left", weight="normal", z=6):
        ax.text(x, y, s, fontsize=size * FS, color=color, ha=ha, va="center",
                zorder=z, weight=weight)

    def elbow(pts, color=LINE, lw=1.4, head=True, z=5, ms=10):
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            ax.add_patch(FancyArrowPatch(
                pts[i], pts[i + 1], arrowstyle="-|>" if (last and head) else "-",
                mutation_scale=ms, linewidth=lw, color=color,
                shrinkA=0, shrinkB=0, joinstyle="round", zorder=z))

    def cells(x0, y0, slots):
        """Twelve token slots; entry is a colour, or None for a discarded slot."""
        for i, c in enumerate(slots):
            if c == "":
                continue
            cx = x0 + 10 + i * 18
            if c is None:
                ax.add_patch(Rectangle((cx, y0 + 9), 14, 20, linewidth=1.0,
                                       edgecolor=ORANGE, facecolor=ORANGE, alpha=0.28,
                                       linestyle=(0, (2, 2)), zorder=5))
            else:
                bar(cx, y0 + 9, 14, 20, c, z=5)

    # ---- panels and phase headers ----
    for x, w, accent, num, title, res in (
        (8, 276, BLUE, "1", "Generate", "GPU"),
        (312, 276, GREEN, "2", "Reconstruct & select", "CPU"),
        (616, 276, ORANGE, "3", "Estimate", "CPU"),
    ):
        card(x, 8, w, 498, r=14, z=1)
        bar(x, 8, w, 5, accent, z=2)
        ax.add_patch(Circle((x + 26, 36), 12.5, facecolor=accent, linewidth=0, zorder=4))
        txt(x + 26, 36.5, num, 12, "white", "center", "bold")
        txt(x + 48, 36.5, title, 12.8, INK, weight="bold")
        txt(x + w - 16, 36.5, res, 9.5, MUTED, ha="right")

    # ---- prompt ----
    card(62, 62, 176, 56, r=28)
    ax.add_patch(Circle((90, 90), 10, facecolor="#EEF1F5", linewidth=0, zorder=4))
    ax.plot([85, 95], [90, 90], color=MUTED, lw=1.4, zorder=5)
    ax.plot([90, 90], [85, 95], color=MUTED, lw=1.4, zorder=5)
    txt(112, 84, "Problem $x_i$", 12.5, INK, weight="bold")
    txt(112, 104, "zero-shot prompt", 11, MUTED)

    # trunk down the left gutter, clear of every label
    elbow([(150, 118), (150, 132), (16, 132), (16, 369)], head=False)
    elbow([(16, 231), (26, 231)])
    elbow([(16, 369), (26, 369)])

    # ---- three arms, one token axis, cap always at local x = 120 ----
    def arm(x0, y, label, lcol, title, note, slots, cut_colour,
            cap_label=None, foot=()):
        txt(x0, y, label, 10.5, lcol, weight="bold")
        txt(x0, y + 20, title, 12.5, INK, weight="bold")
        txt(x0, y + 37, note, 11, MUTED)
        by = y + 50
        card(x0, by, 240, 38, r=8, ec=lcol, fc=lcol + "14", lw=1.3, z=3)
        cells(x0, by, slots)
        ax.plot([x0 + 120, x0 + 120], [by + 3, by + 35],
                color=cut_colour, lw=1.5, ls=(0, (3, 2)), zorder=6)
        if cap_label:
            txt(x0 + 130, by + 19, cap_label, 10, MUTED)
        for fx, s in foot:
            txt(x0 + fx, by + 56, s, 10, MUTED, ha="center")
        return by

    arm(28, 162, "ARM A", BLUE, "Independent draw", "seed $s_1$ · base cap B",
        [BLUE] * 6 + [""] * 6, BLUE, cap_label="cap B")
    arm(28, 300, "ARM C", ORANGE, "Extended trajectory", "seed $s_2$ · cap 2B",
        [GREEN] * 6 + [ORANGE] * 6, ORANGE,
        foot=((62, "shared prefix"), (178, "added tokens")))
    arm(338, 206, "ARM B", GREEN, "Exact base-cap prefix",
        "recovered from arm C token ids",
        [GREEN] * 6 + [None] * 6, ORANGE,
        foot=((178, "discarded beyond the cap"),))

    txt(338, 336, "chains that ended naturally are", 10, MUTED)
    txt(338, 350, "token-identical in arms B and C", 10, MUTED)

    # ---- scoring: the instrument behind every correct/wrong judgment ----
    card(332, 54, 240, 40, r=10, ec=LINE, fc="#F6F7F9", lw=1.1)
    txt(452, 68, "Verify each chain", 12.5, INK, "center", "bold")
    txt(452, 85, "extract answer, compare, then vote", 10.5, MUTED, "center")

    # ---- eligibility gate ----
    ax.add_patch(Polygon([(354, 108), (534, 108), (556, 140), (534, 172),
                          (354, 172), (332, 140)], closed=True,
                         facecolor="#EAF3EC", edgecolor=GREEN, linewidth=1.3, zorder=3))
    txt(444, 132, "CBW eligibility", 12.5, INK, "center", "bold")
    txt(444, 154, "evaluated on arm A only", 11, MUTED, "center")
    elbow([(452, 94), (452, 106)], LINE, 1.35)

    card(332, 376, 240, 54, r=27, ec=GREEN, fc="#EAF3EC", lw=1.3)
    ax.plot([354, 361, 375], [402, 409, 392], color=GREEN, lw=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)
    txt(388, 396, "Exact reconstruction", 12.5, INK, weight="bold")
    txt(388, 415, "zero additional inference", 11, MUTED)

    # ---- flows ----
    elbow([(268, 231), (298, 231), (298, 74), (328, 74)], BLUE, 1.7)
    elbow([(268, 369), (298, 369), (298, 275), (334, 275)], GREEN, 1.7)
    elbow([(556, 140), (594, 140), (594, 121), (623, 121)], LINE, 1.35)
    elbow([(578, 275), (598, 275), (598, 225), (623, 225)], GREEN, 1.7)
    elbow([(148, 388), (148, 452), (606, 452), (606, 329), (623, 329)], ORANGE, 1.7)

    # ---- the three contrasts ----
    def estimate(y, accent, tag, tag_x, name, note, icon):
        card(628, y, 248, 88, r=13)
        bar(628, y, 5, 88, accent, z=4)
        icon(y)
        txt(tag_x, y + 26, tag, 12.5, INK, weight="bold")
        txt(646, y + 52, name, 13, INK, weight="bold")
        txt(646, y + 73, note, 10.5, MUTED)

    def icon_ab(y):
        ax.add_patch(Circle((652, y + 26), 8.5, facecolor=BLUE, linewidth=0, zorder=5))
        elbow([(665, y + 26), (690, y + 26)], LINE, 1.3)
        ax.add_patch(Circle((704, y + 26), 8.5, facecolor=GREEN, linewidth=0, zorder=5))

    def icon_bc(y):
        bar(644, y + 18, 46, 17, GREEN, 4, z=5)
        elbow([(696, y + 26), (716, y + 26)], LINE, 1.3)
        bar(722, y + 18, 32, 17, GREEN, 4, z=5)
        bar(756, y + 18, 26, 17, ORANGE, 4, z=5)

    def icon_ac(y):
        ax.add_patch(Circle((652, y + 26), 8.5, facecolor=BLUE, linewidth=0, zorder=5))
        elbow([(665, y + 26), (690, y + 26)], LINE, 1.3)
        bar(698, y + 18, 32, 17, GREEN, 4, z=5)
        bar(732, y + 18, 26, 17, ORANGE, 4, z=5)

    estimate(74, BLUE, "A $\\rightarrow$ B", 726, "Resampling",
             "same budget · different trajectory", icon_ab)
    estimate(178, GREEN, "B $\\rightarrow$ C", 800, "Continuation",
             "same prefix · added tokens only", icon_bc)
    estimate(282, ORANGE, "A $\\rightarrow$ C", 776, "Combined",
             "trajectory change + token increment", icon_ac)

    # ---- result bus: the contrasts are parallel, not chained ----
    for yy in (118, 222, 326):
        elbow([(876, yy), (884, yy)], LINE, 1.3, head=False)
    elbow([(884, 118), (884, 386), (752, 386), (752, 404)], LINE, 1.3, ms=8)

    card(638, 410, 228, 84, r=17)
    txt(752, 436, "Paired attribution", 13, INK, "center", "bold")
    txt(752, 460, "repair · reversal · exact bounds", 10.5, MUTED, "center")
    txt(752, 478, "paired tests · routing utility", 10.5, MUTED, "center")

    save(fig, "three_arm_protocol")
    plt.close(fig)


EQUIV_OUTPUTS = ROOT / "outputs" / "equivalence"


def make_incremental_bounds():
    """Forest plot of the one-sided upper bound on incremental (B->C) repair."""
    df = pd.read_csv(EQUIV_OUTPUTS / "equivalence_power.csv")
    df = df[df["contrast"] == "B_to_C_incremental_repair"].copy()

    label = {"core_3seed": "core split, 3 seeds",
             "r1_3seed": "core split, 3 seeds",
             "full_split": "full split, 1 seed",
             "core_T0.60_3seed": "core, $T$=0.60",
             "core_T0.75_3seed": "core, $T$=0.75",
             "core_T1.00_3seed": "core, $T$=1.00"}
    # core_3seed is the T=0.75 operating point, so the T=0.75 sweep cells hold
    # the same records and are not drawn twice.
    order = ["core_3seed", "r1_3seed", "full_split",
             "core_T0.60_3seed", "core_T1.00_3seed"]
    name = {"qwen": "Qwen", "llama": "Llama", "r1": "R1"}

    rows = []
    for scope in order:
        for model in ("qwen", "llama", "r1"):
            hit = df[(df["scope"] == scope) & (df["model"] == model)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append((f"{name[model]} — {label[scope]}",
                         int(r["n"]), int(r["k"]),
                         float(r["ci95_upper_onesided"]),
                         float(r["detectable_rate_power80"])))

    rows.reverse()

    fig, ax = plt.subplots(figsize=(8.4, 0.46 * len(rows) + 1.5))
    y = range(len(rows))

    for margin, style in ((0.10, ":"), (0.05, "--")):
        ax.axvline(margin * 100, color="#B0B0B0", linestyle=style, linewidth=1.1, zorder=0)
        ax.text(margin * 100, len(rows) - 0.35, f"{int(margin*100)}%",
                ha="center", va="bottom", fontsize=8, color="#808080")

    for i, (lab, n, k, up, mde) in zip(y, rows):
        colour = "#C44E52" if k else "#4C72B0"
        ax.plot([0, up * 100], [i, i], color=colour, linewidth=2.6,
                solid_capstyle="round", zorder=2)
        ax.plot([up * 100], [i], marker="|", markersize=10, color=colour, zorder=3)
        ax.plot([k / n * 100], [i], marker="o", markersize=5.5, color=colour,
                zorder=4, clip_on=False)
        ax.plot([mde * 100], [i], marker="d", markersize=4.6,
                color="#8C8C8C", zorder=3)
        ax.text(up * 100 + 0.45, i, f"{up*100:.1f}%  ({k}/{n})",
                va="center", fontsize=8.2, color="#333333")

    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("incremental repair rate within the complete-but-wrong subset (%)")
    ax.set_xlim(-0.4, 15.5)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EDEDED", zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#4C72B0", marker="o", markersize=5.5, lw=2.6,
               label="observed rate and 95% one-sided upper bound"),
        Line2D([], [], color="#C44E52", marker="o", markersize=5.5, lw=2.6,
               label="cell with one observed event"),
        Line2D([], [], color="#8C8C8C", marker="d", markersize=4.6, lw=0,
               label="rate at which one event becomes 80% likely"),
    ], loc="lower right", fontsize=8.2, frameon=False)

    save(fig, "incremental_bounds")


def main():
    make_three_arm_routing()
    make_answer_churn()
    make_three_arm_protocol()
    make_incremental_bounds()
    print(f"Saved paper figures to {FIGURES}")


if __name__ == "__main__":
    main()
