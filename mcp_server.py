import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from mcp.server.fastmcp import FastMCP
from app.services import (
    load_index,
    search_chunks,
    rerank_chunks,
    generate_answer_stream,
)

mcp = FastMCP("AutoRAG-问界")

UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")


@mcp.tool()
def semantic_search(query: str, car_model: str = "general", top_k: int = 10) -> list[str]:
    """加载索引，检索并重排序，返回 top3 文档片段。"""
    index, chunks = load_index(car_model)
    if index is None:
        return {"error": f"未找到 {car_model} 的索引"}
    results = search_chunks(query, index, chunks, top_k=top_k)
    return rerank_chunks(query, chunks, top_k=3)


@mcp.tool()
def rag_qa(question: str, car_model: str = "general") -> str:
    """检索相关文档片段并生成完整回答。"""
    chunks = semantic_search(question, car_model)
    if not chunks:
        return "未找到相关索引，请先上传文档并建立索引。"
    return "".join(generate_answer_stream(question, chunks))


@mcp.tool()
def list_documents() -> list[str]:
    """返回 data/uploads/ 目录下所有文件名。"""
    if not os.path.exists(UPLOADS_DIR):
        return []
    return [f for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))]


@mcp.resource("rag://stats")
def rag_stats() -> str:
    """返回知识库统计信息的 JSON 字符串。"""
    doc_count = 0
    if os.path.exists(UPLOADS_DIR):
        doc_count = len([
            f for f in os.listdir(UPLOADS_DIR)
            if os.path.isfile(os.path.join(UPLOADS_DIR, f))
        ])
    stats = {
        "document_count": doc_count,
        "embedding_model": "BAAI/bge-small-zh",
        "reranker_model": "BAAI/bge-reranker-base",
        "llm_model": "glm-4",
        "index_type": "FAISS IndexFlatL2",
    }
    return json.dumps(stats, ensure_ascii=False)


@mcp.prompt()
def car_qa_prompt(question: str, car_model: str = "问界M8") -> str:
    """生成车辆售后问答的提示模板。"""
    return (
        f"你是一名专业的{car_model}售后工程师，请回答以下问题：\n\n"
        f"车型：{car_model}\n"
        f"问题：{question}\n\n"
        f"请根据{car_model}的官方资料，给出准确、简洁的回答。"
        f"如果涉及安全操作，请特别提醒用户注意安全。"
    )


if __name__ == "__main__":
    mcp.run()
