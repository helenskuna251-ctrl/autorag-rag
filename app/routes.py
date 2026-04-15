from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import traceback

from app.services import (
    parse_file,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    search_chunks,
    save_index,
    load_index,
    rerank_chunks,
    generate_answer_stream,
)

router = APIRouter()
indexes = {}  # 内存缓存

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 允许的车型白名单(防止前端瞎传)
ALLOWED_CAR_MODELS = {"m9ev", "m8", "s800evr"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), car_model: str = None):
    """上传 PDF 并构建索引,car_model 必填"""
    if not car_model or car_model not in ALLOWED_CAR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"car_model 必填且必须在 {ALLOWED_CAR_MODELS} 中"
        )
    try:
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        text = parse_file(filepath)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        vector_index = build_faiss_index(embeddings)
        save_index(vector_index, chunks, car_model)
        indexes[car_model] = (vector_index, chunks)

        return {
            "filename": file.filename,
            "car_model": car_model,
            "status": "index built",
            "chunks": len(chunks),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str


@router.post("/query/stream")
async def query_stream(query: QueryRequest, car_model: str = None):
    """流式查询,car_model 必填"""
    if not car_model or car_model not in ALLOWED_CAR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"car_model 必填且必须在 {ALLOWED_CAR_MODELS} 中"
        )

    if car_model in indexes:
        vector_index, chunks = indexes[car_model]
    else:
        vector_index, chunks = load_index(car_model)
        if vector_index is None:
            raise HTTPException(404, f"未找到 {car_model} 的索引,请先上传")
        indexes[car_model] = (vector_index, chunks)

    results = search_chunks(query.query, vector_index, chunks, top_k=10)
    results = rerank_chunks(query.query, results, top_k=3)

    return StreamingResponse(
        generate_answer_stream(query.query, results),
        media_type="text/plain"
    )