from .dependencies import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .features import *  # noqa: F401,F403
from .matching import *  # noqa: F401,F403

def summarize(values: list[float]) -> str:
    tensor = torch.tensor(values, dtype=torch.float64)
    return (
        f"mean={float(tensor.mean()):.6e} "
        f"median={float(tensor.median()):.6e} "
        f"p90={float(torch.quantile(tensor, 0.9)):.6e} "
        f"max={float(tensor.max()):.6e}"
    )


def summary_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred = ["split", "name", "block", "function_check"]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_barycenter_convergence(
    rows: list[dict[str, float]], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    iterations = [int(row["iteration"]) for row in rows]
    axis.plot(iterations, [row["mean"] for row in rows], marker="o", label="Mean")
    axis.plot(iterations, [row["median"] for row in rows], marker="s", label="Median")
    axis.fill_between(
        iterations,
        [row["p10"] for row in rows],
        [row["p90"] for row in rows],
        alpha=0.2,
        label="10–90 percentile",
    )
    axis.set_xlabel("Barycenter iteration")
    axis.set_ylabel("Matched assignment cost")
    axis.set_title("Barycenter alignment convergence")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "barycenter_convergence")


def plot_identity_vs_matched(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    splits = row_splits(rows)
    if not splits:
        return
    figure, axes = plt.subplots(
        1, len(splits), figsize=(max(4.2, 3.8 * len(splits)), 4.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    for axis, split in zip(axes, splits):
        current = [row for row in rows if row["split"] == split]
        for row in current:
            axis.plot(
                [0, 1],
                [row["identity_cost"], row["matched_cost"]],
                color=colors.get(split, "#4C78A8"),
                alpha=0.16,
                linewidth=0.8,
            )
        identity = [row["identity_cost"] for row in current]
        matched = [row["matched_cost"] for row in current]
        axis.scatter(
            [0] * len(identity), identity, s=8, alpha=0.35, color=colors.get(split, "#4C78A8")
        )
        axis.scatter(
            [1] * len(matched), matched, s=8, alpha=0.35, color=colors.get(split, "#4C78A8")
        )
        axis.plot(
            [0, 1],
            [np.median(identity), np.median(matched)],
            color="black",
            marker="D",
            linewidth=2.0,
            label="Median",
        )
        axis.set_xticks([0, 1], ["Identity", "Matched"])
        axis.set_title(f"{split} (n={len(current)})")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Assignment cost (lower is better)")
    figure.suptitle("Per-task cost before and after hidden-neuron permutation")
    save_figure(figure, output_dir, "identity_vs_matched_cost")


def plot_block_improvement(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    splits = row_splits(rows)
    if not splits:
        return
    figure, axes = plt.subplots(
        1, len(splits), figsize=(max(4.2, 4.0 * len(splits)), 4.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, split in zip(axes, splits):
        values = [
            [
                row["improvement"]
                for row in rows
                if row["split"] == split and row["block"] == block
            ]
            for block in range(4)
        ]
        axis.boxplot(values, tick_labels=[f"B{block}" for block in range(4)], showfliers=False)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(split)
        axis.set_xlabel("Residual block")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Identity cost − matched cost")
    figure.suptitle("Alignment improvement by block and held-out split")
    save_figure(figure, output_dir, "block_improvement")


def plot_function_equivalence(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    splits = row_splits(rows)
    if not splits:
        return
    values = [
        [max(row["max_abs"], 1.0e-12) for row in rows if row["split"] == split]
        for split in splits
    ]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.boxplot(values, tick_labels=splits, showfliers=True)
    axis.set_yscale("log")
    axis.set_ylabel("Max |f(original) − f(permuted)|")
    axis.set_title("Functional equivalence after permutation")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output_dir, "function_equivalence")


def plot_improvement_heatmap(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    task_names = sorted({f"{row['split']}/{row['name']}" for row in rows})
    lookup = {
        (f"{row['split']}/{row['name']}", int(row["block"])): row["improvement"]
        for row in rows
    }
    matrix = np.asarray(
        [[lookup[(task, block)] for block in range(4)] for task in task_names]
    )
    height = max(5.0, min(14.0, 0.045 * len(task_names)))
    figure, axis = plt.subplots(figsize=(6.2, height))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(4), [f"Block {block}" for block in range(4)])
    if len(task_names) <= 50:
        axis.set_yticks(range(len(task_names)), task_names, fontsize=6)
    else:
        axis.set_yticks([])
        axis.set_ylabel(f"Tasks sorted by split/name (n={len(task_names)})")
    axis.set_title("Per-task, per-block assignment improvement")
    figure.colorbar(image, ax=axis, label="Cost improvement")
    save_figure(figure, output_dir, "task_block_improvement_heatmap")


def build_statistical_summary(
    task_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    function_tolerance: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}, "blocks": {}}
    for split in row_splits(task_rows):
        current = [row for row in task_rows if row["split"] == split]
        identity = np.asarray([row["identity_cost"] for row in current])
        matched = np.asarray([row["matched_cost"] for row in current])
        improvement = identity - matched
        if np.allclose(improvement, 0.0):
            wilcoxon_statistic = 0.0
            wilcoxon_pvalue = 1.0
        else:
            test = wilcoxon(
                identity, matched, alternative="greater", method="auto"
            )
            wilcoxon_statistic = float(test.statistic)
            wilcoxon_pvalue = float(test.pvalue)
        summary["splits"][split] = {
            "tasks": len(current),
            "identity_cost": summary_stats(identity.tolist()),
            "matched_cost": summary_stats(matched.tolist()),
            "improvement": summary_stats(improvement.tolist()),
            "mean_relative_improvement": float(
                np.mean(improvement / np.maximum(np.abs(identity), 1.0e-12))
            ),
            "wilcoxon_identity_greater_statistic": wilcoxon_statistic,
            "wilcoxon_identity_greater_pvalue": wilcoxon_pvalue,
            "function_pass_rate": float(
                np.mean([row["max_abs"] <= function_tolerance for row in current])
            ),
            "max_abs": summary_stats([row["max_abs"] for row in current]),
            "rel_mse": summary_stats([row["rel_mse"] for row in current]),
        }
        summary["blocks"][split] = {
            str(block): summary_stats(
                [
                    row["improvement"]
                    for row in block_rows
                    if row["split"] == split and row["block"] == block
                ]
            )
            for block in range(4)
        }
    return summary


def write_analysis_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    device_label: str,
    reference_record: TaskRecord,
    frozen_hash: str,
    barycenter_rows: list[dict[str, float]],
    task_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    permutations: dict[str, list[list[int]]],
) -> None:
    write_csv(output_dir / "barycenter_iterations.csv", barycenter_rows)
    write_csv(output_dir / "task_metrics.csv", task_rows)
    write_csv(output_dir / "block_metrics.csv", block_rows)
    statistics = build_statistical_summary(
        task_rows, block_rows, args.function_tolerance
    )
    payload = {
        "config": {
            "cost_type": args.cost_type,
            "weight_ratio": args.weight_ratio,
            "activation_ratio": args.activation_ratio,
            "bias_scale": args.bias_scale,
            "iterations": args.iterations,
            "activation_samples": args.activation_samples,
            "activation_sample_seed": args.activation_sample_seed,
            "feature_cache": args.feature_cache,
            "feature_cache_enabled": getattr(args, "feature_cache_enabled", None),
            "probe_samples": args.probe_samples,
            "probe_seed": args.probe_seed,
            "function_tolerance": args.function_tolerance,
            "devices": device_label,
            "data_root": str(args.data_root),
            "dataset": args.dataset,
            "scenario": args.scenario,
            "task_root": str(args.task_root),
            "splits": args.splits,
            "adapter_name": args.adapter_name,
            "save_aligned": args.save_aligned,
            "aligned_name": args.aligned_name,
            "reference_task": reference_record.name,
            "reference_sha256": frozen_hash,
        },
        "statistics": statistics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with gzip.open(output_dir / "permutations.json.gz", "wt", encoding="utf-8") as file:
        json.dump(permutations, file, ensure_ascii=False)
    if not args.no_plots:
        plot_barycenter_convergence(barycenter_rows, output_dir)
        plot_identity_vs_matched(task_rows, output_dir)
        plot_block_improvement(block_rows, output_dir)
        plot_function_equivalence(task_rows, output_dir)
        plot_improvement_heatmap(block_rows, output_dir)

