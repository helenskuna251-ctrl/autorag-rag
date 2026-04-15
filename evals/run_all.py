"""
evals/run_all.py — AutoRAG Ragas 全量评测入口

用法:
    python evals/run_all.py --model m9ev --testset evals/testset_m9ev_final.json

输出:
    - reports/{model}_{timestamp}.csv   题级明细
    - stdout                             汇总统计
"""

# ============================================================
# Monkey patch 区:必须在 import ragas 之前生效
# ------------------------------------------------------------
# Ragas 0.2.x 的 LLM/Embedding contract 和我们的裁判配置不匹配,
# evals/judge_config.py 里打了补丁修掉这个问题。
# 升级到 Ragas 0.3+ 后可以评估是否移除。
# ============================================================
from evals.judge_config import ragas_judge_llm, ragas_judge_emb  # noqa: E402,F401

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.services import load_index, search_chunks, generate_answer_stream


# ============================================================
# 配置区
# ============================================================

# Refusal 关键词表 — 命中任一即判为"模型正确拒答"
# 翻 bad case 发现漏判,往这里加词即可
REFUSAL_KEYWORDS = [
    # 显式拒答
    "无法回答", "无法提供", "无法", "抱歉",
    # 车型架构澄清(陷阱题核心场景)
    "不配备", "不具备", "没有", "不支持", "不适用",
    "纯电车型", "纯电版", "增程版", "非本车型",
    # 功能/部件不存在
    "不存在", "未配备",
]


# ============================================================
# 核心函数
# ============================================================

def load_testset(path: Path) -> list[dict]:
    """读 testset JSON"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_rag(question: str, index, chunks, top_k: int = 5) -> tuple[str, list[str]]:
    """对单条问题跑 RAG: 检索 + 生成"""
    retrieved = search_chunks(question, index, chunks, top_k=top_k)
    answer = "".join(generate_answer_stream(question, retrieved))
    return answer, retrieved


def check_refusal(should_refuse: bool, answer: str) -> bool | None:
    """
    Refusal Accuracy 判定
    返回:
      - None:  非陷阱题,不参与该指标
      - True:  陷阱题 且 模型正确拒答
      - False: 陷阱题 但 模型没有拒答(该拒没拒,产生了幻觉)
    """
    if not should_refuse:
        return None
    hit = any(kw in answer for kw in REFUSAL_KEYWORDS)
    return hit


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["m9ev", "m8", "s800evr"],
                        help="车型(对应索引名)")
    parser.add_argument("--testset", required=True,
                        help="testset JSON 路径")
    parser.add_argument("--top_k", type=int, default=5,
                        help="检索 top_k(默认 5)")
    args = parser.parse_args()

    # ----- 1. 加载 -----
    print(f"[1/5] 加载索引: {args.model}")
    index, chunks = load_index(args.model)
    if index is None:
        print(f"  ❌ 索引加载失败,检查 {args.model}_index.faiss 是否存在")
        return
    print(f"  已加载 {len(chunks)} 个 chunks")

    print(f"\n[2/5] 加载 testset: {args.testset}")
    testset = load_testset(Path(args.testset))
    print(f"  共 {len(testset)} 条题")

    # ----- 2. 跑 RAG -----
    print(f"\n[3/5] 跑 RAG pipeline (top_k={args.top_k})")
    questions, answers, contexts_list, ground_truths = [], [], [], []
    refusal_results = []

    for i, item in enumerate(testset, 1):
        q = item["user_input"]
        gt = item.get("reference", "")
        print(f"  [{i}/{len(testset)}] {q[:50]}...")

        answer, retrieved = run_rag(q, index, chunks, args.top_k)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(retrieved if retrieved else ["(无检索结果)"])
        ground_truths.append(gt)
        refusal_results.append(
            check_refusal(item.get("should_refuse", False), answer)
        )

    # ----- 3. Ragas 四指标 -----
    print(f"\n[4/5] Ragas 评测 (裁判: DeepSeek, 可能 5-10 分钟)")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_judge_llm,
        embeddings=ragas_judge_emb,
    )
    df = result.to_pandas()

    # ----- 4. 写题级明细 CSV -----
    print(f"\n[5/5] 写报表")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = Path(f"reports/{args.model}_{ts}.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "hop_type", "should_refuse", "refusal_correct",
            "question", "answer_preview",
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ])
        for i, item in enumerate(testset):
            w.writerow([
                i,
                item.get("hop_type", ""),
                item.get("should_refuse", False),
                refusal_results[i] if refusal_results[i] is not None else "",
                item["user_input"],
                answers[i][:300].replace("\n", " "),
                f"{df.iloc[i]['faithfulness']:.3f}",
                f"{df.iloc[i]['answer_relevancy']:.3f}",
                f"{df.iloc[i]['context_precision']:.3f}",
                f"{df.iloc[i]['context_recall']:.3f}",
            ])

    # ----- 5. 汇总 stdout -----
    print("\n" + "=" * 60)
    print(f"  {args.model} baseline ({ts})")
    print("=" * 60)
    print(f"  样本数           : {len(testset)}")
    print(f"  Faithfulness     : {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy : {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")
    print(f"  Context Recall   : {df['context_recall'].mean():.3f}")

    refusal_valid = [r for r in refusal_results if r is not None]
    if refusal_valid:
        correct = sum(refusal_valid)
        total = len(refusal_valid)
        pct = correct / total
        print(f"  Refusal Accuracy : {correct}/{total} ({pct:.1%})")
    else:
        print(f"  Refusal Accuracy : N/A (testset 中无陷阱题)")
    print("=" * 60)
    print(f"\n  明细已写入: {report_path}")


if __name__ == "__main__":
    main()
