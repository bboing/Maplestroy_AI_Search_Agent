FROM python:3.13-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libxml2-dev \
    libxslt-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 한글 locale 설정
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Python 의존성 설치 (통합 requirements.txt 사용)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 애플리케이션 코드 복사
COPY langchain_app /app/langchain_app
COPY scripts /app/scripts

# langchain_app을 Python path에 추가
ENV PYTHONPATH="/app/langchain_app:${PYTHONPATH}"

# Docker 환경변수 자동 설정 (scripts 실행용)
# ENV로 설정하면 모든 실행에서 자동 적용!
ENV BIZ_POSTGRES_HOST=biz-postgres \
    MILVUS_HOST=milvus \
    REDIS_HOST=redis \
    NEO4J_HOST=neo4j

# 헬스체크 엔드포인트
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FastAPI 서버 실행
WORKDIR /app/langchain_app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
