from fastapi import FastAPI,Request
from app.routes import router
import logging
logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"收到请求: {request.method} {request.url}")
    response = await call_next(request)
    return response
app.include_router(router)