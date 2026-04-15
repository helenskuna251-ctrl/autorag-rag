import fitz
import faiss
import numpy as np
import re
import pickle
import logging
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# ---------- 模型加载 ----------
from sentence_transformers import SentenceTransformer, CrossEncoder
model = SentenceTransformer('BAAI/bge-small-zh')
reranker = CrossEncoder("BAAI/bge-reranker-base")


# ---------- PDF 解析 ----------
def read_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(filepath, skip_pages=3):
    """读取 PDF,跳过前 N 页(封面+目录污染区,默认3)"""
    doc = fitz.open(filepath)
    text = ""
    for i, page in enumerate(doc):
        if i < skip_pages:
            continue
        text += page.get_text()
    return text

def parse_file(filepath, skip_pages=3):
    logger.info(f"开始解析文件:{filepath}")
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        return read_txt(filepath)
    elif ext == '.pdf':
        return read_pdf(filepath, skip_pages=skip_pages)
    else:
        raise ValueError(f'Unsupported file type: {ext}')


# ---------- Chunk 切分(新版,RecursiveCharacterTextSplitter) ----------
def chunk_text(text, chunk_size=500, chunk_overlap=80):
    """
    用 LangChain 的 RecursiveCharacterTextSplitter 切分
    优先在段落/换行/句号/逗号处切,极少从句中断
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "!", "?", ";", ",", " ", ""],
        length_function=len,  # 按字符数计算长度,中文友好
    )
    chunks = splitter.split_text(text)
    logger.info(f"切分完成,共 {len(chunks)} 个 chunks")
    return chunks


# ---------- Embedding 与索引 ----------
def create_embeddings(chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    logger.info(f"开始构建索引,向量数量:{len(embeddings)}")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 改用内积(IP),配合 normalize_embeddings 等价余弦相似度
    index.add(embeddings)
    logger.info("索引构建完成")
    return index


# ---------- 检索(简化版,无父子结构) ----------
def search_chunks(question, index, chunks, top_k=10):
    """
    向量检索,返回 top_k 个最相关的 chunk
    简化:不再有父子结构,直接返回 chunk 文本列表
    """
    question_vector = model.encode([question], normalize_embeddings=True)
    question_vector = np.array(question_vector).astype("float32")
    distances, indices = index.search(question_vector, top_k)
    results = []
    for i in indices[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[int(i)])
    return results


# ---------- Reranker ----------
def rerank_chunks(query, chunks, top_k=3):
    if not chunks:
        return []
    pairs = [(query, c) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]


# ---------- GLM-4 答题 ----------
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("ZHIPU_API_KEY 未找到")

from zhipuai import ZhipuAI
client = ZhipuAI(api_key=api_key)

def generate_answer_stream(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""你是一名专业的汽车售后工程师。
请根据以下参考资料回答用户问题,回答要简洁准确,使用中文回答。
如果资料中没有相关信息,请直接说明无法从资料中找到答案。

参考资料:
{context}

用户问题:{question}

回答:"""
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def clen_answer(text):
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------- 索引保存/加载(简化版) ----------
def save_index(index, chunks, car_model):
    """保存索引和 chunks,按 car_model 分桶"""
    if not car_model or car_model == "general":
        raise ValueError("car_model 必须显式指定(m9ev/m8/s800evr 等),不允许 'general' 默认值")
    logger.info(f"保存索引到磁盘,car_model={car_model}")
    faiss.write_index(index, f"data/{car_model}_index.faiss")
    with open(f"data/{car_model}_chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks}, f)
    logger.info(f"索引保存完成,共 {len(chunks)} 个 chunks")

def load_index(car_model):
    """加载指定车型的索引"""
    index_path = f"data/{car_model}_index.faiss"
    chunks_path = f"data/{car_model}_chunks.pkl"
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, []
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"]