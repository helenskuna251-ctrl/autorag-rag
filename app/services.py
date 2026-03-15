import fitz
import faiss
import numpy as np
import re
import pickle
import logging
from dotenv import load_dotenv
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 指定 .env 文件路径
env_path = os.path.join(BASE_DIR, ".env")

# 加载 .env
load_dotenv(env_path)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
def read_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
def read_pdf(filepath):
    doc = fitz.open(filepath)

    text=""

    for page in doc:
        text += page.get_text()
    return text


def parse_file(filepath):
    logger.info(f"开始解析文件：{filepath}")
    _,ext = os.path.splitext(filepath)
    if ext == '.txt':
        return read_txt(filepath)
    elif ext == '.pdf':
        return read_pdf(filepath)
    else:
        logger.error(f"不支持的文件类型：{ext}")
        raise ValueError('Unsupported file type')

def chunk_text_parent_child(text):

    lines = text.split("\n")

    parent_chunks = []
    child_chunks = []
    child_to_parent = {}
    current_parent = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue


        is_title = (
                len(line) < 25 and  # 更短
                not line.startswith('●') and
                not line.startswith('注') and
                not line[-1] in ['。', '，', '、', '；', '：'] and  # 不以标点结尾
                (re.match(r'^[\d一二三四五六七八九十]+[\.、]', line) or  # 数字/中文数字开头
                 re.match(r'^第.+[章节]', line))  # 或者"第X章/节"格式
        )

        if is_title and current_parent :
            parent_chunks.append(current_parent.strip())
            current_parent_idx = len(parent_chunks) - 1
            current_parent = line + "\n"

        elif is_title:
            current_parent = line + "\n"
        else:
            current_parent += line + "\n"

            child_idx = len(child_chunks)
            child_chunks.append(line)
            child_to_parent[child_idx] = max(0,len(parent_chunks))

    if current_parent :
        parent_chunks.append(current_parent.strip())

    return parent_chunks,child_chunks,child_to_parent

from sentence_transformers import SentenceTransformer
model=SentenceTransformer('BAAI/bge-small-zh')

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    return embeddings

def build_faiss_index(embeddings):
    logger.info(f"开始构建索引，向量数量：{len(embeddings)}")
    dimnsion=embeddings.shape[1]
    index = (faiss.IndexFlatL2(dimnsion))
    index.add(embeddings)
    logger.info("索引构建完成")
    return index

def search_chunks(question,index,child_chunks,parent_chunks,child_to_parent,top_k:10):
    question_vector = model.encode([question])
    distances, indices = index.search(question_vector, top_k)
    results = []
    seen_parents = set()
    for i in indices[0]:
        if i <len(child_chunks):
            parent_idx = child_to_parent.get(i,0)
            if parent_idx not in seen_parents:
                seen_parents.add(parent_idx)
                results.append(parent_chunks[parent_idx])
    return results

api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("ZHIPU_API_KEY 未找到，请检查 .env 文件")
print("读取到API_KEY:", api_key[:10], "...")

from zhipuai import ZhipuAI
client = ZhipuAI(api_key=api_key)
def generate_answer_stream(question,chunks):
    context = "\n\n".join(chunks)
    prompt = f"""你是一名专业的汽车售后工程师。
    请根据以下参考资料回答用户问题，回答要简洁准确，使用中文回答。
    如果资料中没有相关信息，请直接说明无法从资料中找到答案。
    
    参考资料:
    {context}
    
    用户回答问题:{question}


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
    text = re.sub(r"\*+","",text)
    text = re.sub(r"\n+","\n",text)
    text = re.sub(r"\s+"," ",text)
    text = text.strip()
    return text

def rerank_chunks(query,chunks,top_k=3):
    """
    使用 reranker 对检索到的 chunks 重新排序。

    参数:
        query (str): 用户问题
        chunks (list): vector search 返回的文本
        top_k (int): 最终返回的 chunk 数量

    返回:
        list: rerank 后的最相关 chunks
    """
    pairs = [(query,chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks,scores),key=lambda x:x[1],reverse=True)
    reranked_chunks = [chunk for chunk,score in ranked[:top_k]]
    return reranked_chunks

INDEX_PATH = 'data/faiss_index.index'
def save_index(index,child_chunks,parent_chunks,child_to_parent,car_model):
    logger.info("开始保存索引到磁盘")
    faiss.write_index(index, f"data/{car_model}_index.faiss")

    with open(f"data/{car_model}_chunks.pkl","wb") as f:
        pickle.dump({
            "child_chunks":child_chunks,
            "parent_chunks":parent_chunks,
            "child_to_parent":child_to_parent,
        },f)
    logger.info(f"索引保存完成，共{len(child_chunks)}个子chunk,{len(parent_chunks)}个父chunk")

def load_index(car_model):
    index_path = f"data/{car_model}_index.faiss"
    chunks_path = f"data/{car_model}_chunks.pkl"
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None,[],[],{}
    index = faiss.read_index(index_path)
    with open(chunks_path,"rb") as f:
        data = pickle.load(f)
        child_chunks = data["child_chunks"]
        parent_chunks = data["parent_chunks"]
        child_to_parent = data["child_to_parent"]
        return index,child_chunks,parent_chunks,child_to_parent



