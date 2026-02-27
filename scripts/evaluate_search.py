"""
검색 시스템 평가 스크립트

MRR, nDCG@K, Precision@K, Recall@K 계산
"""
import sys
import json
import asyncio
import time
import numpy as np
import importlib
from pathlib import Path
from typing import List, Dict, Any


# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "langchain_app"))

from database.session import get_async_db

def calculate_mrr(results: List[Dict], ground_truth: List[str]) -> float:
    """
    MRR (Mean Reciprocal Rank) 계산
    
    첫 번째 정답의 순위 역수
    
    Args:
        results: 검색 결과 리스트
        ground_truth: 정답 엔티티 이름 리스트
        
    Returns:
        MRR 점수 (0~1)
    """
    for rank, result in enumerate(results, start=1):
        name = result.get("data", {}).get("canonical_name", "")
        if name in ground_truth:
            return 1.0 / rank
    return 0.0


def calculate_ndcg(
    results: List[Dict],
    relevance: Dict[str, int],
    k: int = 10
) -> float:
    """
    nDCG@K (Normalized Discounted Cumulative Gain) 계산
    
    순위를 고려한 정확도
    
    Args:
        results: 검색 결과 리스트
        relevance: 엔티티별 관련도 {이름: 점수(0-3)}
        k: 상위 K개까지 평가
        
    Returns:
        nDCG 점수 (0~1)
    """
    # DCG 계산
    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        name = result.get("data", {}).get("canonical_name", "")
        rel = relevance.get(name, 0)
        if rel > 0:
            dcg += rel / np.log2(rank + 1)
    
    # IDCG 계산 (이상적 순서)
    ideal_rels = sorted(relevance.values(), reverse=True)
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels[:k], start=1) if rel > 0)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(
    results: List[Dict],
    ground_truth: List[str],
    k: int = 5
) -> float:
    """
    Precision@K 계산
    
    상위 K개 중 정답 비율
    
    Args:
        results: 검색 결과 리스트
        ground_truth: 정답 엔티티 이름 리스트
        k: 상위 K개
        
    Returns:
        Precision 점수 (0~1)
    """
    top_k = [r.get("data", {}).get("canonical_name", "") for r in results[:k]]
    hits = len(set(top_k) & set(ground_truth))
    return hits / k if k > 0 else 0.0


def calculate_recall_at_k(
    results: List[Dict],
    ground_truth: List[str],
    k: int = 10
) -> float:
    """
    Recall@K 계산
    
    전체 정답 중 상위 K개에 포함된 비율
    
    Args:
        results: 검색 결과 리스트
        ground_truth: 정답 엔티티 이름 리스트
        k: 상위 K개
        
    Returns:
        Recall 점수 (0~1)
    """
    top_k = [r.get("data", {}).get("canonical_name", "") for r in results[:k]]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


async def evaluate_single_query(
    searcher: "HybridSearcher",
    test_case: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, float]:
    """
    단일 질문 평가
    
    Args:
        searcher: HybridSearcher 인스턴스
        test_case: 테스트 케이스
        verbose: 상세 출력 여부
        
    Returns:
        평가 메트릭
    """
    query = test_case["query"]
    ground_truth = test_case["ground_truth"]
    relevance = test_case["relevance"]
    print(f"relevance:{relevance}")
    print(f"ground_truth:{ground_truth}")
    # 검색 실행 + 응답시간 측정
    try:
        print("evaluate_search.evaluate_single_query 호출 됨")
        t_start = time.perf_counter()
        results = await searcher.search(query, limit=10)
        latency_ms = (time.perf_counter() - t_start) * 1000
        print(f"results: {results}")
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    # 메트릭 계산
    # try:
    #     mrr = calculate_mrr(results, ground_truth)
    #     ndcg_10 = calculate_ndcg(results, relevance, k=10)
    #     ndcg_5 = calculate_ndcg(results, relevance, k=5)
    #     precision_5 = calculate_precision_at_k(results, ground_truth, k=5)
    #     recall_10 = calculate_recall_at_k(results, ground_truth, k=10)
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     raise
    mrr = calculate_mrr(results, ground_truth)
    ndcg_10 = calculate_ndcg(results, relevance, k=10)
    ndcg_5 = calculate_ndcg(results, relevance, k=5)
    precision_5 = calculate_precision_at_k(results, ground_truth, k=5)
    recall_10 = calculate_recall_at_k(results, ground_truth, k=10)
    
    metrics = {
        "mrr": mrr,
        "ndcg@10": ndcg_10,
        "ndcg@5": ndcg_5,
        "precision@5": precision_5,
        "recall@10": recall_10,
        "latency_ms": latency_ms
    }
    
    if verbose:
        print(f"\n질문: {query}")
        print(f"정답: {ground_truth}")
        print(f"결과 ({len(results)}개):")
        for i, r in enumerate(results[:5], 1):
            name = r.get("data", {}).get("canonical_name", "Unknown")
            score = r.get("score", 0)
            is_correct = "✅" if name in ground_truth else "  "
            print(f"  {i}. {name} ({score:.2f}) {is_correct}")
        print(f"메트릭: MRR={mrr:.3f}, nDCG@10={ndcg_10:.3f}, P@5={precision_5:.3f}, 응답시간={latency_ms:.0f}ms")
    
    return metrics


async def evaluate_search_system(
    test_file: str = "training/data/test/search_test_queries.json",
    verbose: bool = True,
    option: int = 0
) -> Dict[str, Any]:
    """
    검색 시스템 전체 평가
    
    Args:
        test_file: 테스트 파일 경로
        verbose: 상세 출력 여부
        option: 0=현재(Plan), 2=임계값, 3=Intent, 4=완전병렬
        
    Returns:
        평가 결과
    """
    import importlib

    module_configs = {
        0: {"path": "src.retrievers.hybrid_searcher", "name": "현재 (Plan 기반)"},
        2: {"path": "src.retrievers.hybrid_searcher_option2", "name": "Option 2 (임계값 기반)"},
        3: {"path": "src.retrievers.hybrid_searcher_option3", "name": "Option 3 (Intent 기반)"},
        4: {"path": "src.retrievers.hybrid_searcher_option4", "name": "Option 4 (완전 병렬 + 키워드)"},
        5: {"path": "src.retrievers.hybrid_searcher_sep", "name": "sep (Plan + 키워드, 문장 분류 + 동의어 서치)"},
        6: {"path": "src.retrievers.hybrid_searcher_hop", "name": "hop (쿼리 깊이 분류 적용)"},
        7: {"path": "src.retrievers.hybrid_searcher_fin", "name": "fin (HOP 구조 + PG canonical_name → Neo4j 보강)"}
    }
    
    config = module_configs.get(option, module_configs[0])
    option_name = config["name"]

    # 2. 동적 임포트 실행 (빨간 줄/캐싱 방지 핵심)
    # import_module은 "src.retrievers.hybrid_searcher" 같은 문자열을 받음
    module = importlib.import_module(config["path"])
    importlib.reload(module)  # 혹시 모를 캐시 방지
    
    # 해당 파일 안에서 'HybridSearcher'라는 클래스를 꺼내옴
    TargetSearcherClass = getattr(module, "HybridSearcher")

    # 3. 테스트 케이스 로드
    test_path = ROOT_DIR / test_file
    with open(test_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    print(f"{'='*60}")
    print(f"검색 시스템 평가 시작: {option_name}")
    print(f"테스트 케이스: {len(test_cases)}개")
    print(f"{'='*60}")
    
    all_metrics = []
    
    # 4. DB 세션 안에서 동적으로 가져온 클래스로 객체 생성
    async for session in get_async_db():
        searcher = TargetSearcherClass(
            db=session,
            use_milvus=True,
            use_neo4j=True,
            use_router=True,
            verbose=False
        )
        print(f"DEBUG: [{option_name}] 클래스 위치 -> {searcher.__class__.__module__}")
        
        for i, test_case in enumerate(test_cases, 1):
            if verbose: print(f"\n[{i}/{len(test_cases)}] ", end="")
            try:
                metrics = await evaluate_single_query(searcher, test_case, verbose=verbose)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"❌ 평가 실패: {e}")
                all_metrics.append({"mrr": 0.0, "ndcg@10": 0.0, "ndcg@5": 0.0, "precision@5": 0.0, "recall@10": 0.0, "latency_ms": 0.0})
        break
    
    if not all_metrics:
        print("❌ 평가된 케이스가 없습니다")
        return {}

    # 평균 계산 (latency는 별도 통계)
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in all_metrics[0].keys()
    }

    latencies = [m["latency_ms"] for m in all_metrics]
    latency_stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"평가 결과 요약")
    print(f"{'='*60}")
    print(f"MRR (Mean Reciprocal Rank):      {avg_metrics['mrr']:.4f}")
    print(f"nDCG@10:                          {avg_metrics['ndcg@10']:.4f}")
    print(f"nDCG@5:                           {avg_metrics['ndcg@5']:.4f}")
    print(f"Precision@5:                      {avg_metrics['precision@5']:.4f}")
    print(f"Recall@10:                        {avg_metrics['recall@10']:.4f}")
    print(f"--- 응답시간 ---")
    print(f"평균:  {latency_stats['mean_ms']:.0f}ms")
    print(f"중간값: {latency_stats['median_ms']:.0f}ms")
    print(f"P95:   {latency_stats['p95_ms']:.0f}ms")
    print(f"최소:  {latency_stats['min_ms']:.0f}ms  /  최대: {latency_stats['max_ms']:.0f}ms")
    print(f"{'='*60}")
    
    # 카테고리별 분석
    category_metrics = {}
    for test_case, metrics in zip(test_cases, all_metrics):
        category = test_case["category"]
        if category not in category_metrics:
            category_metrics[category] = []
        category_metrics[category].append(metrics["mrr"])
    
    print(f"\n카테고리별 MRR:")
    for category, mrrs in sorted(category_metrics.items()):
        avg_mrr = np.mean(mrrs)
        print(f"  {category:20s}: {avg_mrr:.4f}")
    
    return {
        "average": avg_metrics,
        "latency": latency_stats,
        "individual": all_metrics,
        "by_category": category_metrics
    }


async def compare_systems():
    """
    여러 시스템 비교 (현재 vs Option 2 vs Option 3 vs Option 4 vs sep vs hop vs fin)
    """
    print("\n" + "="*85)
    print("검색 시스템 비교: 현재(Plan) vs Option 2(임계값) vs Option 3(Intent) vs Option 4(완전병렬) vs sep(Plan + 키워드, 문장 분류 + 동의어 서치) vs hop(쿼리 깊이 분류 적용) vs fin(HOP 구조 + PG canonical_name → Neo4j 보강)")
    print("="*85)
    
    # 현재 (Plan) 평가
    print("\n[1/7] 현재 시스템 (Plan 기반) 평가 중...")
    current_results = await evaluate_search_system(verbose=False, option=0)
    
    # Option 2 평가
    print("\n[2/7] Option 2 (임계값 기반) 평가 중...")
    option2_results = await evaluate_search_system(verbose=False, option=2)
    
    # Option 3 평가
    print("\n[3/7] Option 3 (Intent 기반) 평가 중...")
    option3_results = await evaluate_search_system(verbose=False, option=3)
    
    # Option 4 평가
    print("\n[4/7] Option 4 (완전 병렬 + 키워드) 평가 중...")
    option4_results = await evaluate_search_system(verbose=False, option=4)

    # sep 평가
    print("\n[5/7] Option 5 (Plan + 키워드, 문장 분류 + 동의어 서치) 평가 중...")
    sep_results = await evaluate_search_system(verbose=False, option=5)

    # hop 평가
    print("\n[6/7] Option 6 (쿼리 깊이 분류 적용) 평가 중...")
    hop_results = await evaluate_search_system(verbose=False, option=6)

    # fin 평가
    print("\n[7/7] Option 7 (HOP 구조 + PG canonical_name → Neo4j 보강) 평가 중...")
    fin_results = await evaluate_search_system(verbose=False, option=7)

    
    # 비교표 출력
    print(f"\n{'='*125}")
    print(f"비교 결과")
    print(f"{'='*125}")
    print(f"{'메트릭':<15s} {'현재(Plan)':<15s} {'Option 2':<15s} {'Option 3':<15s} {'Option 4':<15s} {'sep':<15s} {'hop':<15s} {'fin':<15s}")
    print(f"{'-'*125}")

    metrics = ["mrr", "ndcg@10", "ndcg@5", "precision@5", "recall@10"]
    for metric in metrics:
        current_val = current_results["average"][metric]
        opt2_val = option2_results["average"][metric]
        opt3_val = option3_results["average"][metric]
        opt4_val = option4_results["average"][metric]
        sep_val = sep_results["average"][metric]
        hop_val = hop_results["average"][metric]
        fin_val = fin_results["average"][metric]
        print(f"{metric:<15s} {current_val:<15.4f} {opt2_val:<15.4f} {opt3_val:<15.4f} {opt4_val:<15.4f} {sep_val:<15.4f} {hop_val:<15.4f} {fin_val:<15.4f}")

    # 응답시간 비교 (mean / p95)
    print(f"{'-'*125}")
    all_results = [current_results, option2_results, option3_results, option4_results, sep_results, hop_results, fin_results]
    means = "  ".join(f"{r['latency']['mean_ms']:<13.0f}" for r in all_results)
    p95s  = "  ".join(f"{r['latency']['p95_ms']:<13.0f}" for r in all_results)
    print(f"{'latency(mean)':<15s} {means}")
    print(f"{'latency(p95)':<15s} {p95s}")
    print(f"{'='*125}")

    # ── 응답시간 핵심 요약 (Plan 1 / 최저 / 최고 / Plan 7) ──────────────
    all_named = [
        ("Plan 1 (현재)",    current_results["latency"]["mean_ms"],  current_results["latency"]["p95_ms"]),
        ("Option 2",         option2_results["latency"]["mean_ms"],  option2_results["latency"]["p95_ms"]),
        ("Option 3",         option3_results["latency"]["mean_ms"],  option3_results["latency"]["p95_ms"]),
        ("Option 4",         option4_results["latency"]["mean_ms"],  option4_results["latency"]["p95_ms"]),
        ("sep",              sep_results["latency"]["mean_ms"],      sep_results["latency"]["p95_ms"]),
        ("hop",              hop_results["latency"]["mean_ms"],      hop_results["latency"]["p95_ms"]),
        ("Plan 7 / fin",     fin_results["latency"]["mean_ms"],      fin_results["latency"]["p95_ms"]),
    ]
    fastest = min(all_named, key=lambda x: x[1])
    slowest = max(all_named, key=lambda x: x[1])
    plan1   = all_named[0]
    plan7   = all_named[-1]

    print(f"\n{'─'*55}")
    print(f"  응답시간 요약 (mean / P95)")
    print(f"{'─'*55}")
    print(f"  Plan 1 (현재)   : {plan1[1]:>7.0f}ms  /  P95 {plan1[2]:>7.0f}ms")
    print(f"  최저 시간 Plan  : {fastest[1]:>7.0f}ms  /  P95 {fastest[2]:>7.0f}ms  ← {fastest[0]}")
    print(f"  최고 시간 Plan  : {slowest[1]:>7.0f}ms  /  P95 {slowest[2]:>7.0f}ms  ← {slowest[0]}")
    print(f"  Plan 7 (fin)    : {plan7[1]:>7.0f}ms  /  P95 {plan7[2]:>7.0f}ms")
    diff = plan7[1] - plan1[1]
    sign = "+" if diff >= 0 else ""
    print(f"{'─'*55}")
    print(f"  Plan 7 - Plan 1 : {sign}{diff:.0f}ms  ({'느림' if diff > 0 else '빠름'})")
    
    # 최고 성능 표시
    systems = {
        "현재 (Plan 기반)": current_results["average"],
        "Option 2 (임계값 기반)": option2_results["average"],
        "Option 3 (Intent 기반)": option3_results["average"],
        "Option 4 (완전 병렬 + 키워드)": option4_results["average"],
        "sep (Plan + 키워드, 문장 분류 + 동의어 서치)": sep_results["average"],
        "hop (쿼리 깊이 분류 적용)": hop_results["average"],
        "fin (HOP 구조 + PG canonical_name → Neo4j 보강)": fin_results["average"],
    }

    best_name = max(systems, key=lambda name: (
        systems[name]["mrr"],
        systems[name]["ndcg@10"],
        systems[name]["ndcg@5"],
        systems[name]["precision@5"],
        systems[name]["recall@10"]
    ))

    print(f"\n🏆 최고 성능: {best_name}")

    # 비교 결과 저장
    output_dir = ROOT_DIR / "training/data/output_data"
    output_file_path = output_dir / "evaluation_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_data = {
        "metrics": metrics,
        "current": {**current_results["average"], "latency": current_results["latency"]},
        "option2": {**option2_results["average"], "latency": option2_results["latency"]},
        "option3": {**option3_results["average"], "latency": option3_results["latency"]},
        "option4": {**option4_results["average"], "latency": option4_results["latency"]},
        "sep":     {**sep_results["average"],     "latency": sep_results["latency"]},
        "hop":     {**hop_results["average"],     "latency": hop_results["latency"]},
        "fin":     {**fin_results["average"],     "latency": fin_results["latency"]},
    }
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=4, ensure_ascii=False)
    print("\n✅ 평가 보고서가 'evaluation_report.json'에 저장되었습니다.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="검색 시스템 평가")
    parser.add_argument("--mode", choices=["single", "compare"], default="single",
                        help="평가 모드 (single: 단일 평가, compare: 시스템 비교)")
    parser.add_argument("--option", type=int, choices=[0, 2, 3, 4, 5, 6, 7], default=0,
                        help="검색 옵션 (0: 현재(Plan), 2: 임계값, 3: Intent, 4: 완전병렬, 5: 키워드 문장 분리, 6: 쿼리 깊이 분류, 7: HOP 구조 + PG canonical_name → Neo4j 보강)")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        asyncio.run(compare_systems())
    else:
        asyncio.run(evaluate_search_system(verbose=args.verbose, option=args.option))
