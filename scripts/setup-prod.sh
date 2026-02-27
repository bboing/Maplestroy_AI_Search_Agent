#!/bin/bash
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'


# 1. 에러 발생 시 즉시 중단 및 루트 경로 설정
set -e
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
ROOT_DIR="$( cd "$( dirname "$SCRIPT_PATH" )/.." && pwd )"

echo "=========================================="
echo "🚀 AI Platform Setup (Portfolio Mode)"
echo "📂 Root: $ROOT_DIR"
echo "=========================================="

# [함수] 서비스 대기 로직
wait_for_service() {
    local host="$1"
    local port="$2"
    local name="$3"
    local timeout=60
    local counter=0

    echo -n "   ⏳ $name ($port) 연결 확인 중"
    while ! nc -z "$host" "$port" > /dev/null 2>&1; do
        sleep 1
        echo -n "."
        ((counter++))
        if [ $counter -gt $timeout ]; then
            echo -e "\n   ❌ $name 대기 시간 초과 (60초)"
            exit 1
        fi
    done
    echo -e "\n   ✅ $name 연결 성공!"
}

# 1️⃣ 사전 의존성 및 Docker 상태 확인
echo -e "\n1️⃣ 시스템 환경 확인 중..."

if ! command -v python3 &> /dev/null; then echo "❌ Python3 미설치"; exit 1; fi
if ! command -v docker &> /dev/null; then echo "❌ Docker 미설치"; exit 1; fi

# Docker 실행 여부 확인
if ! docker info &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   🐳 Docker Desktop 실행 중... (macOS)"
        open -a Docker
        sleep 20
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "   ❌ Docker가 실행 중이지 않습니다. (Linux)"
        echo "   👉 아래 명령어로 Docker를 시작하세요:"
        echo "      sudo systemctl start docker"
        exit 1
    else
        echo "   ❌ Docker가 실행 중이지 않습니다."
        echo "   👉 Windows 사용자: Docker Desktop을 먼저 실행하고 WSL2 터미널에서 재시도하세요."
        exit 1
    fi
fi
echo "   ✅ Python & Docker 준비 완료"

# 2️⃣ 환경 변수(.env) 설정
echo -e "\n2️⃣ 환경 변수 설정..."
if [ ! -f "${ROOT_DIR}/.env" ]; then
    if [ -f "${ROOT_DIR}/.env.example" ]; then
        cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
        echo "   📋 .env 파일 생성 완료 (GROQ_API_KEY를 설정하세요!)"
    else
        echo "   ❌ .env.example 파일이 없습니다."; exit 1
    fi
else
    echo "   ✅ .env 파일 확인 완료"
fi

# GROQ_API_KEY 확인
if ! grep -q "^GROQ_API_KEY=gsk_" "${ROOT_DIR}/.env" 2>/dev/null; then
    echo -e "\n   ⚠️  ${YELLOW}GROQ_API_KEY가 설정되지 않았습니다!${NC}"
    echo "   https://console.groq.com/ 에서 API Key를 발급받아 .env에 추가하세요:"
    echo "   ${YELLOW}GROQ_API_KEY=gsk_your_key_here${NC}"
    echo ""
    read -p "   계속 진행하시겠습니까? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        exit 1
    fi
fi

# 3️⃣ 인프라 가동 (Docker Compose - Portfolio Mode)
echo -e "\n3️⃣ Docker 컨테이너 가동 (Portfolio Mode)..."
docker-compose -f "${ROOT_DIR}/docker-compose.prod.yml" up -d --build

wait_for_service "localhost" 5432 "Postgres"
wait_for_service "localhost" 7687 "Neo4j"
wait_for_service "localhost" 19530 "Milvus"

echo "   ⏳ Reranker 모델 로딩 대기 중... (60초)"
sleep 60
wait_for_service "localhost" 8001 "Reranker"

echo "   ⏳ 서비스 안정화 대기 중... (10초)"
sleep 10

# 4️⃣ 가상환경 및 라이브러리 설치
echo -e "\n4️⃣ Python 가상환경 설정..."
if [ ! -d "${ROOT_DIR}/nexon_venv" ]; then
    python3 -m venv "${ROOT_DIR}/nexon_venv"
fi
source "${ROOT_DIR}/nexon_venv/bin/activate"

echo "   의존성 라이브러리 설치 중..."
pip install --upgrade pip > /dev/null
# 통합 requirements.txt 사용 (langchain_app + streamlit_app + scripts 전체 포함)
pip install -r "${ROOT_DIR}/requirements.txt" > /dev/null

# 5️⃣ 데이터베이스 초기화 및 시딩 (중요 순서!)
echo -e "\n5️⃣ 데이터 지식 구조화 시작..."

echo "   [1/3] Postgres 데이터 임포트..."
echo "ROOT_DIR : ${ROOT_DIR}"
python3 "${ROOT_DIR}/scripts/import_data.py" "${ROOT_DIR}/training/data/input_data/maple_data.json"

echo "   [2/3] Milvus 벡터 인덱싱 (기존 데이터 clear 후 재구축)..."
python3 "${ROOT_DIR}/scripts/sync_to_milvus.py" --drop

echo "   [3/3] Neo4j 그래프 관계 구축 (기존 데이터 clear 후 재구축)..."
python3 "${ROOT_DIR}/scripts/sync_to_neo4j.py" || {
    echo "   ❌ Neo4j 동기화 실패"
    exit 1
}

echo "   ✅ 모든 지식 베이스 구축 완료!"


echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}🎉 Portfolio Platform Setup Completed Successfully${NC}"
echo -e "${GREEN}==========================================================${NC}"

echo -e "\n${CYAN}1. Streamlit 데모 앱 접속${NC}"
echo -e "   브라우저에서 아래 URL로 접속하세요:"
echo -e "   ${YELLOW}http://localhost:8501${NC}"
echo -e "   * Groq API를 사용하므로 Ollama 설치가 필요 없습니다."

echo -e "\n${CYAN}2. 데이터셋 및 프로젝트 참고사항${NC}"
echo -e "   * ${YELLOW}Dataset Location:${NC} training/data/input_data/maple_data.json"
echo -e "   * 본 데이터는 RAG 파이프라인의 ${YELLOW}기술적 검증을 위해 구성된 샘플 데이터셋${NC}입니다."
echo -e "   * 현재 메이플 게임 내 정보와 차이가 있을 수 있으며, 아키텍처 테스트 용도로 최적화되어 있습니다."

echo -e "\n${CYAN}3. Infra 서비스 접속 정보 (Local)${NC}"
echo -e "   ┌────────────┬─────────────────────────────┬────────────────────────────────┐"
echo -e "   │ Service    │ Endpoint                    │ Credentials                    │"
echo -e "   ├────────────┼─────────────────────────────┼────────────────────────────────┤"
echo -e "   │ PostgreSQL │ localhost:5432              │ .env 설정 참조                 │"
echo -e "   │ Milvus UI  │ http://localhost:8081       │ 인증 없음                      │"
echo -e "   │ Neo4j UI   │ http://localhost:7474       │ neo4j / nexonJjang67!neo4j     │"
echo -e "   │ Streamlit  │ http://localhost:8501       │ 인증 없음                      │"
echo -e "   └────────────┴─────────────────────────────┴────────────────────────────────┘"
echo -e "   * 상세 접속 정보는 프로젝트 루트의 .env 파일에서 관리됩니다."

echo -e "\n${GREEN}감사합니다. 추가 문의는 프로젝트 README.md를 참조해주세요.${NC}"
echo -e "${GREEN}==========================================================${NC}\n"
