"""
evals/run_one.py
============================
端到端跑通 1 条陷阱题,验证 Ragas 评测链路
目标:在"脏的 general 大索引"上让 M9纯电陷阱题爆分,留 baseline 证据
"""

import os
import sys
from datasets import Dataset

# ---------- 第1段:导入你项目的业务函数 + 评测配置 ----------
# judge_config 里已经准备好 ragas_judge_llm 和 ragas_judge_emb
from evals.judge_config import ragas_judge_llm, ragas_judge_emb

# 导入你项目 services.py 里的核心函数(不带 Reranker,走最小链路)
from app.services import load_index, search_chunks, generate_answer_stream


# ---------- 第2段:硬编码 1 条陷阱题 ----------
# 真陷阱题:M9纯电问 S800 增程器(增程器是 S800 独有,M9 没有)
# 如果索引隔离失效,系统会从 S800 手册里捞数据回答
question = "问界M9纯电版的疲劳分神监测是什么?如何使用?"
ground_truth = "疲劳分神监测通过驾驶员状态监测(DMS)摄像头检测驾驶员状态,当监测到疲劳驾驶(如打瞌睡)或分神驾驶(如使用手机)时,会通过提示音和仪表显示屏进行安全提醒。开启方法:进入中控屏设置 > 驾驶 > 场景辅助 > 疲劳分神监测,开启对应开关。"

# ---------- 第3段:走你项目的检索链路 ----------
print("=" * 60)
print(f"问题: {question}")
print("=" * 60)

# load_index("general") 加载那份祖传大索引 ——正是 bug 所在
# 预期:三车数据全混在一起,接下来的检索会跨车型捞数据
print("\n[1/4] 加载索引 (m9ev)...")
index, chunks = load_index("m9ev")
if index is None:
    print("❌ 索引加载失败,检查 data/general_index.faiss 是否存在")
    sys.exit(1)
print(f"  已加载: {len(chunks)} 个 chunks")

# search_chunks 的 top_k 参数因为原代码类型注解写错(top_k:10 不是默认值)
# 所以这里必须显式传 top_k=5
print("\n[2/4] 向量检索 (top_k=5, 不带 Reranker)...")
retrieved_contexts = search_chunks(question, index, chunks, top_k=5)
print(f"  检索到 {len(retrieved_contexts)} 个 context")
for i, ctx in enumerate(retrieved_contexts, 1):
    # 只打印每个 context 前 80 字,看大意
    preview = ctx[:80].replace("\n", " ")
    print(f"  [{i}] {preview}...")


# ---------- 第4段:调用 GLM-4 生成答案 ----------
# generate_answer_stream 是流式生成器,需要拼回字符串
print("\n[3/4] GLM-4 生成答案 (流式拼接)...")
answer_parts = []
for chunk in generate_answer_stream(question, retrieved_contexts):
    answer_parts.append(chunk)
answer = "".join(answer_parts)
print(f"  GLM-4 答案: {answer}")


# ---------- 第5段:喂给 Ragas 打分 ----------
# Ragas 的 Dataset 格式要求固定 4 个字段
# 注意 contexts 必须是 list of list (每条样本的 contexts 是一个 list)
# 即使只有 1 条样本,也要包成 [[...]]
print("\n[4/4] Ragas 裁判打分 (DeepSeek 分析中, 可能 30-90 秒)...")
dataset = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [retrieved_contexts],  # 已经是 list,外层再包一层
    "ground_truth": [ground_truth],
})

# 导入 Ragas 四指标
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # 忠实度:答案有没有幻觉(和 context 对不对得上)
    answer_relevancy,      # 相关性:答案有没有跑题
    context_precision,     # 上下文精度:检索的 context 里有多少是真相关的
    context_recall,        # 上下文召回:回答所需的信息有没有被召回
)

# evaluate 函数:
#   - dataset: 上面构造的 4 字段 Dataset
#   - metrics: 四指标列表
#   - llm: 裁判 LLM (DeepSeek)
#   - embeddings: 计算相似度用的 embedding (项目同款 BGE)
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_judge_llm,
    embeddings=ragas_judge_emb,
)


# ---------- 第6段:打印结果 ----------
print("\n" + "=" * 60)
print("📊 Ragas 评测结果 (baseline, 脏索引, 无Reranker)")
print("=" * 60)
# result 可以转成 pandas DataFrame,看起来清爽
df = result.to_pandas()
print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].to_string(index=False))
print("=" * 60)

# 解读提示
print("\n💡 预期解读:")
print("  - faithfulness 若 <0.5: GLM-4 产生了幻觉(用燃油车数据回答纯电车)")
print("  - context_recall ≈ 0: 正确答案(无发动机)根本没被检索到")
print("  - context_precision 低: 检索到的 context 多是 M8 燃油内容,对 M9纯电无效")
print("\n✅ baseline 采集完成,截图保存这个结果作为 'before' 证据!")