
"""
merge_testset.py — 合并 Ragas 原生清洗题 + 人工补充题

功能:
  1. 读 Ragas 清洗后的 testset(12 条)
  2. 读人工补充的 testset(10 条)
  3. 合并为一份,统一字段,输出到 final testset
  4. 确保所有条目都有 should_refuse 字段

用法:
  python merge_testset.py \
      --ragas  evals/testset_m9ev_clean.json \
      --manual evals/testset_m9ev_manual_10.json \
      --output evals/testset_m9ev_final.json
"""

import argparse
import json
from pathlib import Path

# 最终 testset 的标准字段
STANDARD_FIELDS = [
    "user_input",
    "reference_contexts",
    "reference",
    "synthesizer_name",
    "should_refuse",
    "target_model",
    "hop_type",
]


def normalize_item(item: dict, default_target: str = "m9ev") -> dict:
    """把一条题标准化到统一字段"""
    normalized = {
        "user_input": item.get("user_input", ""),
        "reference_contexts": item.get("reference_contexts", []),
        "reference": item.get("reference", ""),
        "synthesizer_name": item.get("synthesizer_name", "unknown"),
        "should_refuse": item.get("should_refuse", False),
        "target_model": item.get("target_model", default_target),
        "hop_type": item.get("hop_type", "ragas_single_hop"),
    }
    # 保留额外字段(如 design_note, source_chapters)
    for k, v in item.items():
        if k not in normalized:
            normalized[k] = v
    return normalized


def merge(ragas_path: Path, manual_path: Path, output_path: Path) -> None:
    with ragas_path.open("r", encoding="utf-8") as f:
        ragas_items = json.load(f)
    with manual_path.open("r", encoding="utf-8") as f:
        manual_items = json.load(f)

    print(f"[Ragas 原生] {len(ragas_items)} 条")
    print(f"[人工补充] {len(manual_items)} 条")

    merged = []
    for item in ragas_items:
        merged.append(normalize_item(item, default_target="m9ev"))
    for item in manual_items:
        merged.append(normalize_item(item))

    # 统计
    total = len(merged)
    refuse_count = sum(1 for x in merged if x["should_refuse"])
    by_model = {}
    by_type = {}
    for x in merged:
        by_model[x["target_model"]] = by_model.get(x["target_model"], 0) + 1
        by_type[x["hop_type"]] = by_type.get(x["hop_type"], 0) + 1

    print(f"\n[合并后] 共 {total} 条")
    print(f"  - 应拒答: {refuse_count} 条")
    print(f"  - 车型分布: {by_model}")
    print(f"  - 题型分布: {by_type}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", required=True, help="Ragas 清洗后的 testset 路径")
    parser.add_argument("--manual", required=True, help="人工补充的 testset 路径")
    parser.add_argument("--output", required=True, help="合并后输出路径")
    args = parser.parse_args()

    merge(Path(args.ragas), Path(args.manual), Path(args.output))


if __name__ == "__main__":
    main()