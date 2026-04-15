"""
evals/gen_testset.py
============================
用 Ragas 0.2 TestsetGenerator 自动生成评测题
- 生成模型:DeepSeek
- Embedding:本地 BGE-small-zh
- 数据源:从 m9ev 索引取 chunks 作为种子文档
- 图谱持久化:第一次 transform 完缓存到磁盘,后续调参免费复用
"""

import os
import pickle
from pathlib import Path

from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms.default import default_transforms
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.persona import Persona
from evals.judge_config import ragas_judge_llm, ragas_judge_emb


# ---------- 参数 ----------
CAR_MODEL = "m9ev"
NUM_TESTSET = 30                              # 5 → 30
CHUNKS_LIMIT = None                           # 50 → None,用全量 779 chunks
KG_CACHE_PATH = Path(f"evals/kg_cache_{CAR_MODEL}_full.json")   # 加 _full 后缀,和冒烟缓存区分开
OUTPUT_PATH = Path(f"evals/testset_{CAR_MODEL}_30.json")        # 输出文件改名

def load_chunks_as_documents(car_model: str, limit: int = None):
    """加载 chunks → LangChain Document"""
    chunks_path = f"data/{car_model}_chunks.pkl"
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    if limit:
        chunks = chunks[:limit]
        print(f"  [冒烟模式] 仅用前 {limit} 个 chunks(共 {len(data['chunks'])} 个)")
    docs = [
        Document(page_content=c, metadata={"source": car_model, "chunk_id": i})
        for i, c in enumerate(chunks)
    ]
    return docs


def get_or_build_knowledge_graph(docs, llm, embeddings):
    """
    ★ 图谱持久化核心:
    - 如果缓存存在,直接 load → 省时间、省钱
    - 否则 build + transform → 保存到磁盘
    """
    if KG_CACHE_PATH.exists():
        print(f"\n[Step 1-2/3] ✅ 发现图谱缓存,直接加载: {KG_CACHE_PATH}")
        kg = KnowledgeGraph.load(str(KG_CACHE_PATH))
        print(f"  已加载 {len(kg.nodes)} 个节点,{len(kg.relationships)} 条边")
        print(f"  💰 省钱:跳过 transform(默认耗时 3-10 分钟 + 数元成本)")
        return kg

    print(f"\n[Step 1/3] 构建初始知识图谱...")
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )
    print(f"  图谱初始化:{len(kg.nodes)} 个节点")

    print(f"\n[Step 2/3] 应用 transforms(抽 summary/theme/entity,建边)...")
    print(f"  ⚠️  第一次 3-10 分钟 + DeepSeek 调用成本,完成后会缓存,后续复用免费")
    transforms = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)
    apply_transforms(kg, transforms)
    print(f"  图谱增强完成:{len(kg.nodes)} 节点,{len(kg.relationships)} 条边")

    # 保存缓存
    KG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    kg.save(str(KG_CACHE_PATH))
    print(f"  💾 图谱已缓存到 {KG_CACHE_PATH}")

    return kg


def generate_testset(kg, llm, embeddings, testset_size: int):
    """手动指定 persona,绕过自动过滤"""
    print(f"\n[Step 3/3] 生成 {testset_size} 条测试题...")

    personas = [
        Persona(
            name="新手车主",
            role_description="刚提车不久的问界M9纯电版车主,对车辆功能不熟悉,"
                             "经常查阅使用手册,问题偏基础,如'某个功能是什么''怎么开启'",
        ),
        Persona(
            name="技术型车主",
            role_description="对电动车技术感兴趣的问界M9纯电版车主,"
                             "关心具体参数和技术细节,如充电功率、续航、电池容量、"
                             "智驾系统版本、具体配置差异",
        ),
        Persona(
            name="日常使用车主",
            role_description="日常通勤使用问界M9纯电版的车主,关心实用功能,"
                             "如如何设置座椅记忆、空调快速制冷、导航充电规划等日常操作",
        ),
    ]

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )
    # 只用 single_hop,跳过需要图谱边的 multi_hop 两种
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 1.0),  # 100% 都用 single_hop
    ]
    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
    )
    return testset


def main():
    docs = load_chunks_as_documents(CAR_MODEL, limit=CHUNKS_LIMIT)
    kg = get_or_build_knowledge_graph(docs, ragas_judge_llm, ragas_judge_emb)
    testset = generate_testset(kg, ragas_judge_llm, ragas_judge_emb, NUM_TESTSET)

    # 保存结果
    df = testset.to_pandas()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(OUTPUT_PATH, orient="records", force_ascii=False, indent=2)
    print(f"\n✅ 生成完成:{OUTPUT_PATH}")
    print(f"  共 {len(df)} 条,字段:{list(df.columns)}")

    # 预览前 3 条
    print("\n" + "=" * 60)
    print("📋 预览前 3 条:")
    print("=" * 60)
    for i, row in df.head(3).iterrows():
        print(f"\n--- 题 {i+1} ---")
        for col in df.columns:
            val = str(row[col])
            if len(val) > 200:
                val = val[:200] + "..."
            print(f"  {col}: {val}")


if __name__ == "__main__":
    main()