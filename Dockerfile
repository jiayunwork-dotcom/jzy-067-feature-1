# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data/profiles

WORKDIR /app

# 依赖（含构建期跑测试用的 pytest）
COPY requirements.txt ./
RUN pip install -r requirements.txt pytest==8.3.5

# 源码与测试
COPY model ./model
COPY tests ./tests
COPY wsgi.py pytest.ini ./

# 构建阶段即跑一遍自动化测试，验残差达标、入渗率收敛、积水判定、
# 单调走向与多方案并发隔离；任何一条不过镜像就构建失败
RUN python -m pytest

# 容器内持久化位置（可挂卷）
RUN mkdir -p /data/profiles && useradd -r -u 10001 ga && chown -R ga /data
USER ga

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "wsgi:app"]
