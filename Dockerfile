# 第一行：选择基础镜像，用 Python 3.10 的精简版
FROM python:3.10-slim

# 设置工作目录，后续所有命令都在这个目录执行
WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存加速构建
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 把项目所有文件复制进容器
COPY . .

# 创建数据目录
RUN mkdir -p data/uploads

# 暴露端口 8000
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]