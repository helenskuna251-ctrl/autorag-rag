"""
clean_testset.py — 评测集清洗工具

功能:
  1. 从原始 testset 中剔除指定索引的题目(人工筛题淘汰的)
  2. 给保留的每条题补上 should_refuse 字段(默认 False)
  3. 输出为新文件,不覆盖原文件

用法:
  python clean_testset.py \
      --input  evals/testset_m9ev_30.json \
      --output evals/testset_m9ev_clean.json \
      --drop   1,2,6

说明:
  --drop 是 0-indexed 的索引列表,逗号分隔
  例如 "1,2,6" 表示剔除第 2、3、7 题(人类视角从 1 数)
"""

import argparse
import json
from pathlib import Path


def clean_testset(input_path: Path, output_path: Path, drop_indices: set[int]) -> None:
    # 读原文件
    with input_path.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    print(f"[原始] 共 {len(testset)} 条")

    # 剔除 + 补字段
    cleaned = []
    for idx, item in enumerate(testset):
        if idx in drop_indices:
            print(f"  [淘汰] idx={idx}: {item['user_input'][:40]}...")
            continue
        # 补 should_refuse 字段,默认 False(非陷阱题)
        item["should_refuse"] = False
        cleaned.append(item)

    print(f"[清洗后] 共 {len(cleaned)} 条,已补 should_refuse=False")

    # 写新文件(不覆盖原文件)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="清洗 Ragas 生成的 testset")
    parser.add_argument("--input", required=True, help="原始 testset JSON 路径")
    parser.add_argument("--output", required=True, help="清洗后输出路径")
    parser.add_argument(
        "--drop",
        required=True,
        help="要剔除的题目索引,0-indexed,逗号分隔。例如: 1,2,6",
    )
    args = parser.parse_args()

    drop_indices = {int(x.strip()) for x in args.drop.split(",") if x.strip()}
    clean_testset(Path(args.input), Path(args.output), drop_indices)


if __name__ == "__main__":
    main()