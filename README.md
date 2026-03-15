# AutoRAG · 汽车售后智能问答系统

基于 RAG 架构的汽车使用手册智能问答系统，支持问界 M8、问界 M9、尊界 S800 等车型使用手册的智能检索与问答。

## 项目特点

- 多车型知识库隔离，每个车型独立索引存储
- 父子 Chunk 分块策略，提升结构化文档检索准确率
- 基于 BGE-Reranker 重排序，提升检索相关性
- 流式输出，回答实时渲染
- Docker 容器化部署，一键启动

## 技术栈

| 模块 | 技术 |
|------|------|
| 接口层 | FastAPI |
| 向量检索 | FAISS |
| Embedding | BAAI/bge-small-zh |
| Rerank | BAAI/bge-reranker-base |
| LLM | 智谱AI GLM-4 |
| 部署 | Docker |

## 核心设计

### 父子 Chunk 分块策略
使用手册具有明显的层级结构（章节→要点），采用父子 Chunk 策略：
- **子 Chunk**：每个要点，用于向量检索，精准匹配用户问题
- **父 Chunk**：完整小节，检索命中后返回给 LLM，保证上下文完整

相比固定字数切块，父子 Chunk 在结构化文档上检索准确率明显提升。

### 多车型知识库隔离
每个车型独立索引存储，查询时按 car_model 路由，避免不同车型知识互相干扰。

## 系统架构
```
用户请求
    ↓
FastAPI 接口层
    ↓
向量检索（FAISS + 子Chunk）
    ↓
返回父Chunk（完整上下文）
    ↓
Rerank 重排序（BGE-Reranker）
    ↓
LLM 生成回答（GLM-4）
    ↓
流式返回给用户
```

## 快速启动

### 方式一：Docker 启动（推荐）
```bash
docker build -t autorag .
docker run -p 8000:8000 --env-file .env autorag
```

### 方式二：本地启动
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| /upload | POST | 上传 PDF，构建索引 |
| /query/stream | POST | 流式问答 |

## 使用示例

1. 上传车型手册：访问 `http://localhost:8000/docs`，调用 `/upload` 接口，上传 PDF，指定 `car_model=wenjie_m8`

2. 智能问答：调用 `/query/stream` 接口，输入问题，指定 `car_model=wenjie_m8`

## 环境变量

在项目根目录创建 `.env` 文件：
```
ZHIPU_API_KEY=你的智谱AI密钥
```