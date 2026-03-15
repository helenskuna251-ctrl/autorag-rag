from app.services import parse_file
from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi.responses import StreamingResponse
from app.services import generate_answer_stream
import os
from app.services import (
    parse_file,
    chunk_text_parent_child,
    create_embeddings,
    build_faiss_index,
    search_chunks,
    save_index,
    rerank_chunks
)
import  traceback
indexes = {}
from app.services import load_index


router =APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)###文件保存路径

import traceback

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), car_model: str = "general"):
    try:
        print("UPLOAD API 被调用了")
        filepath = os.path.join(UPLOAD_DIR, file.filename)

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        text = parse_file(filepath)
        parent_chunks,child_chunks,child_to_parent = chunk_text_parent_child(text)
        embeddings = create_embeddings(child_chunks)
        vector_index = build_faiss_index(embeddings)
        save_index(vector_index,child_chunks ,parent_chunks,child_to_parent, car_model)
        indexes[car_model] = (vector_index,child_chunks,parent_chunks,child_to_parent)

        return {
            "filename": file.filename,
            "status": "index built",
            "child_chunks": len(child_chunks),
            "parent_chunks": len(parent_chunks)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"上传出错了: {e}")
        raise

from pydantic import BaseModel
class  QueryRequest(BaseModel):
    query: str

@router.post("/query/stream")
async def query_stream(query: QueryRequest, car_model: str = "general"):
    if car_model in indexes:
        vector_index, child_chunks, parent_chunks, child_to_parent = indexes[car_model]
    else:
        vector_index, child_chunks, parent_chunks, child_to_parent = load_index(car_model)
        if vector_index is None:
            return {"error": f"没有找到{car_model}的索引，请先上传文档"}
        indexes[car_model] = (vector_index, child_chunks, parent_chunks, child_to_parent)

    results = search_chunks(
        query.query,
        vector_index,
        child_chunks,
        parent_chunks,
        child_to_parent,
        top_k=10
    )
    results = rerank_chunks(query.query, results, top_k=3)

    return StreamingResponse(
        generate_answer_stream(query.query, results),
        media_type="text/plain"
    )




