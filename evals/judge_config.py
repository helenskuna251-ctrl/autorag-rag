"""
evals/judge_config.py
============================
Ragas 评测的"裁判配置中心"
职责:① 配置 DeepSeek 作为 LLM 裁判  ② 复用项目本地 BGE 作为 embedding
"""
# === Monkey patch: 绕过 langchain-core 0.3.x 的 _format_for_tracing bug ===
# Bug: ragas 0.2.10 调 llm.generate 时,langchain-core 内部期望 message.content,
#      但 PromptValue 经过转换后是 str,导致 AttributeError
# 修复:把 _format_for_tracing 替换成恒等函数,跳过这个无关的 tracing 步骤
import langchain_core.language_models.chat_models as _lc_cm
_lc_cm._format_for_tracing = lambda messages: messages
# === Patch end ===
# 后面是原来的 import os / from dotenv ... 那些
import os
from dotenv import load_dotenv

# ---------- 第1段:加载 .env ----------
# 为什么要定位到项目根目录的 .env?
# 因为 run_one.py 可能从 evals/ 目录下跑,也可能从项目根跑,
# 工作目录不确定,load_dotenv() 不给路径就会找不到。
# 这里用 __file__(当前文件绝对路径)往上跳两级,强制锁定根目录。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY 未在 .env 中找到,请检查变量名")


# ---------- 第2段:配置 DeepSeek LLM 裁判 ----------
# 关键知识点:DeepSeek 的 API 是"OpenAI 兼容"的,意思是它实现了 OpenAI 的
# /v1/chat/completions 接口协议,所以可以直接用 langchain_openai 的 ChatOpenAI 类,
# 只要把 base_url 指向 DeepSeek 的服务器即可。这是业界通用套路。
#
# 谐音记忆:base_url = "贝斯 URL",意思是"基地URL",告诉客户端"请求往哪发"
from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(
    model="deepseek-chat",              # DeepSeek 的模型名,对标 GPT-4 级别
    api_key=DEEPSEEK_API_KEY,           # 认证 key
    base_url="https://api.deepseek.com/v1",  # 核心:把请求改道到 DeepSeek
    temperature=0,                      # ★ 评分必须稳定,temperature=0 让输出最确定
    max_tokens=2048,                    # 裁判有时要输出长的 JSON,留够空间
)
# 为什么 temperature=0 这么重要?
# 如果裁判今天给 0.8 明天给 0.3,你做消融实验时分数波动你分不清是"算法改了"
# 还是"裁判心情变了"。评测的第一性原理是"可复现",temperature=0 是地基。


# ---------- 第3段:复用项目本地的 BGE-small-zh ----------
# ★ 这一段是整个 judge_config.py 最关键的部分,出错会让评测分数全部失真 ★
#
# 为什么不能让 Ragas 用它默认的 OpenAI text-embedding?
# 因为你线上系统用 BGE-small-zh 把 chunks 编码进 FAISS,
# 评测时算 Context Precision/Recall 必须用"同一个向量空间"去算相似度,
# 用 OpenAI embedding 等于"拿尺子量体重",单位不匹配,分数是垃圾。
#
# 做法:直接从 app.services import 已经加载好的那个 model 实例,
# 不要 new 新的,新的会多占一份显存,而且万一权重版本对不上就完蛋。
from app.services import model as bge_model  # 这是 SentenceTransformer 实例

# 但 Ragas 只认 LangChain 风格的 embedding 接口(有 .embed_query / .embed_documents 方法),
# SentenceTransformer 的 API 是 .encode(),不兼容。所以要包一层适配器。
#
# langchain-huggingface 提供了 HuggingFaceEmbeddings,能把 sentence-transformers 模型
# 包装成 LangChain 格式。但它会再 new 一个模型实例(浪费),
# 所以这里我手写一个最小适配器,直接复用 bge_model,零浪费。
from langchain_core.embeddings import Embeddings
from typing import List

class LocalBGEEmbeddings(Embeddings):
    """把项目里那个 SentenceTransformer 实例,包装成 LangChain Embeddings 接口"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量编码文档(chunks),返回 list of float list
        # .tolist() 是因为 bge_model.encode 返回 numpy,LangChain 要的是 Python list
        return bge_model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        # 编码单个 query,返回一个 float list
        return bge_model.encode(text, normalize_embeddings=True).tolist()

# 为什么 normalize_embeddings=True?
# BGE 系列官方文档要求:做相似度检索时必须归一化,否则余弦相似度会偏。
# 你 services.py 里那个 model.encode() 其实 *没* 归一化(这是个潜在小瑕疵,
# 但不影响当前评测,先不动它)。我们在评测这里归一化,保证 Ragas 算分准确。

judge_embeddings = LocalBGEEmbeddings()


# ---------- 第4段:包装成 Ragas 认识的对象 ----------
# Ragas 有自己的 LLM/Embedding wrapper,再套一层才能塞进 evaluate() 函数
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper

ragas_judge_llm = LangchainLLMWrapper(judge_llm)
ragas_judge_emb = LangchainEmbeddingsWrapper(judge_embeddings)

# 导出这两个对象给 run_one.py 用
__all__ = ["ragas_judge_llm", "ragas_judge_emb", "judge_llm", "judge_embeddings"]

if __name__ == "__main__":
    # 自测:直接 python evals/judge_config.py 跑这段,验证配置是否加载成功
    print("✅ DeepSeek key 读取成功:", DEEPSEEK_API_KEY[:10], "...")
    print("✅ BGE 模型复用成功:", type(bge_model).__name__)
    test_vec = judge_embeddings.embed_query("问界M9纯电版的发动机")
    print(f"✅ Embedding 测试成功,向量维度: {len(test_vec)}")
    print("全部就绪,可以进入 run_one.py")