import json
from collections import defaultdict

def compute_delta_view(oracle_combo_json,
                       min_combos_required=8,
                       normalized=True):
    """
    Calculate δ_view using δ_combo
    Args:
        oracle_combo_json: dict, JSON output from Part1
        min_combos_required: The minimum number of successful combos required to retain the sample
        normalized: whether to normalize delta

    Returns:
        oracle_view_dict
    """

    oracle_view = {}

    for sample_id, sample_data in oracle_combo_json.items():
        num_combos = sample_data.get('num_combos_success', 0)

        # 1. Filter samples with too few combos
        if num_combos < min_combos_required:
            continue

        # 2. Accumulator
        view_delta_sum = defaultdict(float)
        view_count = defaultdict(int)

        for combo in sample_data["combos"]:
            delta = combo["delta"]
            used_view = combo["used_views"]

            for v in used_view:
                view_delta_sum[v] += delta
                view_count[v] += 1

        # 3. Calculate the average δ_view
        view_stats = {}
        for v in view_delta_sum:
            mean_delta = view_delta_sum[v] / view_count[v]
            view_stats[v] = {
                "count": view_count[v],
                "delta_mean": mean_delta
            }

        # 4. Optional normalization
        if normalized:
            total = sum(v["delta_mean"] for v in view_stats.values())
            if total > 0:
                for v in view_stats:
                    view_stats[v]["delta_norm"] = (
                        view_stats[v]["delta_mean"] / total
                    )
            else:
                for v in view_stats:
                    view_stats[v]["delta_norm"] = 0.0

        oracle_view[sample_id] = {
            "num_combos_success": num_combos,
            "view": view_stats
        }
    return oracle_view

if __name__ == "__main__":
    with open("../data/oracle_raw.json") as f:
        oracle_combo = json.load(f)

    oracle_view = compute_delta_view(
        oracle_combo_json=oracle_combo,
        min_combos_required=8,
        normalized=True
    )

    with open("../data/oracle_view.json", 'w') as f:
        json.dump(oracle_view, f, indent=2)

    print(f"Saved oracle_view.json with {len(oracle_view)} samples.")