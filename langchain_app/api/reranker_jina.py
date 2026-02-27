from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch
import uvicorn

app = FastAPI(title="Local Jina Reranker (MPS)")

# 1. GPU 디바이스 우선순위: MPS(Mac) > CUDA(NVIDIA) > CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"🚀 Using Device: {device}")

# 2. Jina Reranker v2 모델 로드
# trust_remote_code=True는 Jina의 특수 레이어를 불러오기 위해 필수!
model = CrossEncoder(
    'jinaai/jina-reranker-v2-base-multilingual', 
    device=device, 
    trust_remote_code=True
)

class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    top_n: int = 5

@app.post("/rerank")
async def rerank(request: RerankRequest):
    try:
        # 질문과 문서를 쌍으로 묶어서 점수 계산
        pairs = [[request.query, text] for text in request.texts]
        scores = model.predict(pairs)
        
        # 결과 결합 및 정렬
        results = [
            {"index": i, "text": text, "score": float(score)} 
            for i, (text, score) in enumerate(zip(request.texts, scores))
        ]
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {"results": results[:request.top_n]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)