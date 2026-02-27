#!/bin/bash
# Neo4j 완전 초기화 및 재동기화

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
ROOT_DIR="$( cd "$( dirname "$SCRIPT_PATH" )/.." && pwd )"

echo -e "${YELLOW}=========================================="
echo "🔄 Neo4j 완전 재동기화"
echo "==========================================${NC}"

# 1. Neo4j 컨테이너 내부에서 모든 데이터 삭제
echo -e "\n1️⃣ Neo4j 데이터 완전 삭제..."
docker exec ai-neo4j cypher-shell -u neo4j -p maplestory123! -d neo4j "MATCH (n) DETACH DELETE n"

# 2. Python 스크립트로 재동기화
echo -e "\n2️⃣ PostgreSQL → Neo4j 재동기화..."
cd "${ROOT_DIR}"
source nexon_venv/bin/activate
python3 scripts/sync_to_neo4j.py

echo -e "\n${GREEN}✅ Neo4j 재동기화 완료!${NC}"
echo -e "${GREEN}브라우저에서 http://localhost:7474 확인하세요.${NC}\n"
