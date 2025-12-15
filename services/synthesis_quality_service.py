"""합성 품질 평가 서비스"""
import os
import json
import re
import requests
import traceback
from typing import Dict, List, Optional
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse

from services.database import get_db_connection
from core.gemini_client import get_gemini_client_pool
from core.s3_client import get_logs_s3_image
from config.settings import GEMINI_3_FLASH_MODEL


def load_evaluation_prompt() -> str:
    """
    평가 프롬프트 파일을 로드합니다.
    
    Returns:
        str: 평가 프롬프트 텍스트
    """
    prompt_path = os.path.join(os.getcwd(), "prompts", "synthesis_quality_evaluation.txt")
    abs_prompt_path = os.path.abspath(prompt_path)
    
    print(f"[Quality Evaluation] 프롬프트 경로: {abs_prompt_path}")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"[Quality Evaluation] 프롬프트 로드 완료 (길이: {len(content)} 문자)")
        return content
    except FileNotFoundError:
        print(f"[Quality Evaluation] WARNING: 프롬프트 파일을 찾을 수 없습니다: {abs_prompt_path}")
        # 기본 프롬프트 반환
        return """당신은 AI 이미지 합성 품질 평가 전문가입니다. 제공된 이미지는 인물에 드레스를 합성한 결과입니다.

다음 기준에 따라 합성 품질을 평가해주세요:
1. 의상 자연스러움 (30점)
2. 인물-의상 조화 (30점)
3. 배경 합성 품질 (20점)
4. 전체적인 품질 (20점)

다음 JSON 형식으로 응답해주세요:
{
  "score": 0-100,
  "comment": "상세한 평가 내용",
  "is_success": true 또는 false
}

점수 90점 이상을 성공으로 판단합니다."""
    except Exception as e:
        print(f"[Quality Evaluation] ERROR: 프롬프트 로드 실패: {e}")
        return "평가 프롬프트 로드 실패"


def download_image_from_url(url: str) -> Optional[Image.Image]:
    """
    URL에서 이미지를 다운로드합니다.
    S3 URL인 경우 S3 클라이언트를 사용하고, 그 외에는 requests를 사용합니다.
    
    Args:
        url: 이미지 URL
    
    Returns:
        PIL Image 또는 None (실패 시)
    """
    try:
        parsed_url = urlparse(url)
        
        # S3 URL인 경우 (logs 폴더)
        if 's3' in parsed_url.netloc or '.s3.' in parsed_url.netloc:
            # URL에서 파일명 추출
            # 예: https://marryday2.s3.ap-northeast-2.amazonaws.com/logs/1763702989987_xai-gemini-unified-custom-v4_result.png
            # -> 파일명: 1763702989987_xai-gemini-unified-custom-v4_result.png
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2 and path_parts[0] == 'logs':
                file_name = path_parts[1]
                print(f"[Quality Evaluation] S3에서 이미지 다운로드: {file_name}")
                
                # S3 클라이언트를 사용하여 다운로드
                image_data = get_logs_s3_image(file_name)
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    return image
                else:
                    print(f"[Quality Evaluation] S3 이미지 다운로드 실패: {file_name}")
                    return None
            else:
                print(f"[Quality Evaluation] S3 URL 형식을 파싱할 수 없습니다: {url}")
                return None
        else:
            # 일반 HTTP URL인 경우 requests 사용
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image
            else:
                print(f"[Quality Evaluation] 이미지 다운로드 실패: HTTP {response.status_code}")
                return None
    except Exception as e:
        print(f"[Quality Evaluation] 이미지 다운로드 오류: {e}")
        traceback.print_exc()
        return None


def parse_evaluation_response(response_text: str) -> Dict:
    """
    Gemini 응답에서 평가 결과를 파싱합니다.
    
    Args:
        response_text: Gemini 응답 텍스트
    
    Returns:
        dict: {"score": int, "comment": str, "is_success": bool} 또는 None
    """
    try:
        # JSON 코드 블록에서 추출 시도
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 일반 JSON 객체 찾기
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 전체 텍스트에서 JSON 추출 시도
                json_str = response_text
        
        # JSON 파싱
        result = json.loads(json_str)
        
        # 필수 필드 확인
        if "score" not in result:
            print(f"[Quality Evaluation] 응답에 score가 없습니다: {response_text[:200]}")
            return None
        
        score = int(result.get("score", 0))
        comment = result.get("comment", "")
        is_success = result.get("is_success", score >= 90)
        
        # 점수 범위 확인
        if score < 0 or score > 100:
            print(f"[Quality Evaluation] 점수가 범위를 벗어났습니다: {score}")
            return None
        
        return {
            "score": score,
            "comment": comment,
            "is_success": bool(is_success)
        }
    except json.JSONDecodeError as e:
        print(f"[Quality Evaluation] JSON 파싱 오류: {e}")
        print(f"[Quality Evaluation] 응답 텍스트: {response_text[:500]}")
        return None
    except Exception as e:
        print(f"[Quality Evaluation] 응답 파싱 오류: {e}")
        print(f"[Quality Evaluation] 응답 텍스트: {response_text[:500]}")
        return None


def save_evaluation_result(
    result_log_idx: int,
    quality_score: int,
    quality_comment: str,
    is_success: bool
) -> Optional[int]:
    """
    평가 결과를 데이터베이스에 저장합니다.
    
    Args:
        result_log_idx: result_logs.idx
        quality_score: 평가 점수 (0-100)
        quality_comment: 평가 상세 내용
        is_success: 성공 여부
    
    Returns:
        저장된 레코드의 idx 또는 None (실패 시)
    """
    connection = get_db_connection()
    if connection is None:
        return None
    
    try:
        with connection.cursor() as cursor:
            # 기존 평가 결과가 있는지 확인
            check_query = "SELECT idx FROM synthesis_quality_evaluations WHERE result_log_idx = %s"
            cursor.execute(check_query, (result_log_idx,))
            existing = cursor.fetchone()
            
            if existing:
                # 업데이트
                update_query = """
                UPDATE synthesis_quality_evaluations
                SET quality_score = %s, quality_comment = %s, is_success = %s, evaluated_at = CURRENT_TIMESTAMP
                WHERE result_log_idx = %s
                """
                cursor.execute(update_query, (quality_score, quality_comment, is_success, result_log_idx))
                connection.commit()
                print(f"[Quality Evaluation] 평가 결과 업데이트 완료: result_log_idx={result_log_idx}, score={quality_score}")
                return existing['idx']
            else:
                # 새로 삽입
                insert_query = """
                INSERT INTO synthesis_quality_evaluations (result_log_idx, quality_score, quality_comment, is_success)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (result_log_idx, quality_score, quality_comment, is_success))
                connection.commit()
                evaluation_idx = cursor.lastrowid
                print(f"[Quality Evaluation] 평가 결과 저장 완료: idx={evaluation_idx}, result_log_idx={result_log_idx}, score={quality_score}")
                return evaluation_idx
    except Exception as e:
        print(f"[Quality Evaluation] 평가 결과 저장 오류: {e}")
        traceback.print_exc()
        connection.rollback()
        return None
    finally:
        connection.close()


async def evaluate_synthesis_quality(result_log_idx: int) -> Dict:
    """
    단일 합성 결과 이미지를 평가합니다.
    
    Args:
        result_log_idx: result_logs.idx
    
    Returns:
        dict: {
            "success": bool,
            "result_log_idx": int,
            "quality_score": Optional[int],
            "quality_comment": Optional[str],
            "is_success": Optional[bool],
            "error": Optional[str]
        }
    """
    connection = get_db_connection()
    if connection is None:
        return {
            "success": False,
            "result_log_idx": result_log_idx,
            "error": "DB 연결 실패"
        }
    
    try:
        with connection.cursor() as cursor:
            # result_logs에서 이미지 URL 조회
            select_query = "SELECT result_url, model FROM result_logs WHERE idx = %s"
            cursor.execute(select_query, (result_log_idx,))
            result = cursor.fetchone()
            
            if not result:
                return {
                    "success": False,
                    "result_log_idx": result_log_idx,
                    "error": "result_logs에서 레코드를 찾을 수 없습니다"
                }
            
            result_url = result.get("result_url", "")
            if not result_url:
                return {
                    "success": False,
                    "result_log_idx": result_log_idx,
                    "error": "result_url이 비어있습니다"
                }
            
            # 이미지 다운로드
            print(f"[Quality Evaluation] 이미지 다운로드 시작: {result_url}")
            image = download_image_from_url(result_url)
            if image is None:
                return {
                    "success": False,
                    "result_log_idx": result_log_idx,
                    "error": "이미지 다운로드 실패"
                }
            
            print(f"[Quality Evaluation] 이미지 다운로드 완료: {image.size}")
            
            # 프롬프트 로드
            prompt = load_evaluation_prompt()
            
            # Gemini API 호출
            print(f"[Quality Evaluation] Gemini API 호출 시작: model={GEMINI_3_FLASH_MODEL}")
            pool = get_gemini_client_pool()
            response = await pool.generate_content_with_retry_async(
                model=GEMINI_3_FLASH_MODEL,
                contents=[image, prompt]
            )
            
            # 응답 텍스트 추출
            response_text = ""
            if response.candidates and len(response.candidates) > 0:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
            
            if not response_text:
                return {
                    "success": False,
                    "result_log_idx": result_log_idx,
                    "error": "Gemini 응답이 비어있습니다"
                }
            
            print(f"[Quality Evaluation] Gemini 응답 수신 완료 (길이: {len(response_text)} 문자)")
            
            # 응답 파싱
            evaluation_result = parse_evaluation_response(response_text)
            if evaluation_result is None:
                return {
                    "success": False,
                    "result_log_idx": result_log_idx,
                    "error": "응답 파싱 실패",
                    "raw_response": response_text[:500]
                }
            
            # DB 저장
            evaluation_idx = save_evaluation_result(
                result_log_idx=result_log_idx,
                quality_score=evaluation_result["score"],
                quality_comment=evaluation_result["comment"],
                is_success=evaluation_result["is_success"]
            )
            
            if evaluation_idx is None:
                return {
                    "success": False,
                    "result_log_idx": result_log_idx,
                    "error": "평가 결과 저장 실패"
                }
            
            return {
                "success": True,
                "result_log_idx": result_log_idx,
                "evaluation_idx": evaluation_idx,
                "quality_score": evaluation_result["score"],
                "quality_comment": evaluation_result["comment"],
                "is_success": evaluation_result["is_success"]
            }
    except Exception as e:
        print(f"[Quality Evaluation] 평가 중 오류: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "result_log_idx": result_log_idx,
            "error": str(e)
        }
    finally:
        connection.close()


async def evaluate_batch_synthesis_quality(
    model: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict:
    """
    배치로 합성 결과 이미지를 평가합니다.
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
        limit: 평가 개수 제한 (None이면 제한 없음)
    
    Returns:
        dict: {
            "success": bool,
            "total": int,
            "evaluated": int,
            "failed": int,
            "results": List[Dict]
        }
    """
    connection = get_db_connection()
    if connection is None:
        return {
            "success": False,
            "error": "DB 연결 실패"
        }
    
    try:
        with connection.cursor() as cursor:
            # 디버깅: 전체 result_logs 개수 확인
            debug_query = "SELECT COUNT(*) as total FROM result_logs WHERE result_url IS NOT NULL AND result_url != ''"
            cursor.execute(debug_query)
            total_logs = cursor.fetchone()['total']
            print(f"[Quality Evaluation] 디버깅: result_url이 있는 전체 로그 개수: {total_logs}")
            
            if model:
                debug_model_query = "SELECT COUNT(*) as total FROM result_logs WHERE model = %s AND result_url IS NOT NULL AND result_url != ''"
                cursor.execute(debug_model_query, (model,))
                total_model_logs = cursor.fetchone()['total']
                print(f"[Quality Evaluation] 디버깅: 모델 '{model}'의 result_url이 있는 로그 개수: {total_model_logs}")
                
                # 실제 DB에 저장된 모델명 확인 (LIKE 검색)
                like_model_query = "SELECT DISTINCT model FROM result_logs WHERE model LIKE %s AND result_url IS NOT NULL AND result_url != '' LIMIT 10"
                cursor.execute(like_model_query, (f'%{model}%',))
                matching_models = cursor.fetchall()
                print(f"[Quality Evaluation] 디버깅: '{model}'과 일치하는 모델명들: {[m['model'] for m in matching_models]}")
                
                # success=1이고 평가 안 된 항목 개수 확인
                unevaluated_query = """
                SELECT COUNT(*) as total 
                FROM result_logs rl
                LEFT JOIN synthesis_quality_evaluations sqe ON rl.idx = sqe.result_log_idx
                WHERE rl.model = %s 
                AND rl.result_url IS NOT NULL 
                AND rl.result_url != ''
                AND rl.success = 1
                AND sqe.idx IS NULL
                """
                cursor.execute(unevaluated_query, (model,))
                unevaluated_count = cursor.fetchone()['total']
                print(f"[Quality Evaluation] 디버깅: 모델 '{model}'의 미평가 항목 개수 (success=1): {unevaluated_count}")
            
            # 이미 평가된 항목 개수 확인
            evaluated_count_query = "SELECT COUNT(*) as total FROM synthesis_quality_evaluations"
            cursor.execute(evaluated_count_query)
            already_evaluated = cursor.fetchone()['total']
            print(f"[Quality Evaluation] 디버깅: 이미 평가된 항목 개수: {already_evaluated}")
            
            # 미평가 항목 조회
            # LEFT JOIN으로 평가되지 않은 항목만 선택
            # success가 True인 항목만 평가 (합성 실패한 항목은 제외)
            base_query = """
            SELECT rl.idx, rl.result_url, rl.model
            FROM result_logs rl
            LEFT JOIN synthesis_quality_evaluations sqe ON rl.idx = sqe.result_log_idx
            WHERE rl.result_url IS NOT NULL 
            AND rl.result_url != ''
            AND TRIM(rl.result_url) != ''
            AND rl.success = 1
            AND sqe.idx IS NULL
            """
            
            params = []
            if model:
                # 모델명을 LIKE로 검색 (부분 일치 허용)
                base_query += " AND rl.model LIKE %s"
                params.append(f'%{model}%')
            
            base_query += " ORDER BY rl.created_at DESC"
            
            if limit:
                base_query += " LIMIT %s"
                params.append(limit)
            
            print(f"[Quality Evaluation] 디버깅: 실행할 쿼리: {base_query}")
            print(f"[Quality Evaluation] 디버깅: 쿼리 파라미터: {params}")
            
            cursor.execute(base_query, tuple(params))
            results = cursor.fetchall()
            
            total = len(results)
            evaluated = 0
            failed = 0
            evaluation_results = []
            
            print(f"[Quality Evaluation] 배치 평가 시작: 총 {total}개 항목")
            
            for idx, row in enumerate(results, 1):
                result_log_idx = row['idx']
                print(f"\n[Quality Evaluation] [{idx}/{total}] 평가 중: result_log_idx={result_log_idx}")
                
                result = await evaluate_synthesis_quality(result_log_idx)
                
                if result["success"]:
                    evaluated += 1
                else:
                    failed += 1
                
                evaluation_results.append(result)
            
            print(f"\n[Quality Evaluation] 배치 평가 완료: 총 {total}개, 성공 {evaluated}개, 실패 {failed}개")
            
            return {
                "success": True,
                "total": total,
                "evaluated": evaluated,
                "failed": failed,
                "results": evaluation_results
            }
    except Exception as e:
        print(f"[Quality Evaluation] 배치 평가 오류: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        connection.close()


def get_quality_statistics(
    model: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    성공률 통계를 조회합니다.
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
        start_date: 시작 날짜 (YYYY-MM-DD 형식, None이면 제한 없음)
        end_date: 종료 날짜 (YYYY-MM-DD 형식, None이면 제한 없음)
    
    Returns:
        dict: {
            "success": bool,
            "overall": {
                "total": int,
                "success": int,
                "failed": int,
                "success_rate": float
            },
            "by_model": List[Dict]  # 모델별 통계
        }
    """
    connection = get_db_connection()
    if connection is None:
        return {
            "success": False,
            "error": "DB 연결 실패"
        }
    
    try:
        with connection.cursor() as cursor:
            # 기본 WHERE 조건
            where_conditions = []
            params = []
            
            if model:
                where_conditions.append("rl.model = %s")
                params.append(model)
            
            if start_date:
                where_conditions.append("sqe.evaluated_at >= %s")
                params.append(f"{start_date} 00:00:00")
            
            if end_date:
                where_conditions.append("sqe.evaluated_at <= %s")
                params.append(f"{end_date} 23:59:59")
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # 전체 통계
            overall_query = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN sqe.is_success = 1 THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN sqe.is_success = 0 THEN 1 ELSE 0 END) as failed
            FROM synthesis_quality_evaluations sqe
            INNER JOIN result_logs rl ON sqe.result_log_idx = rl.idx
            {where_clause}
            """
            cursor.execute(overall_query, params)
            overall_result = cursor.fetchone()
            
            total = int(overall_result['total'] or 0)
            success = int(overall_result['success'] or 0)
            failed = int(overall_result['failed'] or 0)
            success_rate = float((success / total * 100) if total > 0 else 0.0)
            
            # 모델별 통계
            by_model_query = f"""
            SELECT 
                rl.model,
                COUNT(*) as total,
                SUM(CASE WHEN sqe.is_success = 1 THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN sqe.is_success = 0 THEN 1 ELSE 0 END) as failed,
                AVG(sqe.quality_score) as avg_score
            FROM synthesis_quality_evaluations sqe
            INNER JOIN result_logs rl ON sqe.result_log_idx = rl.idx
            {where_clause}
            GROUP BY rl.model
            ORDER BY total DESC
            """
            cursor.execute(by_model_query, params)
            by_model_results = cursor.fetchall()
            
            by_model = []
            for row in by_model_results:
                model_total = int(row['total'] or 0)
                model_success = int(row['success'] or 0)
                model_failed = int(row['failed'] or 0)
                model_success_rate = float((model_success / model_total * 100) if model_total > 0 else 0.0)
                avg_score = float(row['avg_score']) if row['avg_score'] else 0.0
                
                by_model.append({
                    "model": row['model'],
                    "total": model_total,
                    "success": model_success,
                    "failed": model_failed,
                    "success_rate": round(model_success_rate, 2),
                    "avg_score": round(avg_score, 2)
                })
            
            return {
                "success": True,
                "overall": {
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "success_rate": round(success_rate, 2)
                },
                "by_model": by_model
            }
    except Exception as e:
        print(f"[Quality Evaluation] 통계 조회 오류: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        connection.close()


def get_unevaluated_results(
    model: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    include_evaluated: bool = False
) -> Dict:
    """
    평가되지 않은 result_logs 조회 (수동 평가용)
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
        limit: 조회할 레코드 수 (None이면 전체 조회)
        offset: 시작 위치 (기본값: 0)
        include_evaluated: 이미 평가된 항목도 포함할지 여부 (기본값: False)
    
    Returns:
        dict: {
            "success": bool,
            "images": List[Dict],
            "total": int
        }
    """
    connection = get_db_connection()
    if connection is None:
        return {
            "success": False,
            "error": "DB 연결 실패",
            "images": [],
            "total": 0
        }
    
    try:
        with connection.cursor() as cursor:
            # 기본 WHERE 조건
            where_conditions = [
                "rl.result_url IS NOT NULL",
                "rl.result_url != ''",
                "TRIM(rl.result_url) != ''",
                "rl.success = 1"  # 성공한 항목만
            ]
            params = []
            
            if model:
                where_conditions.append("rl.model LIKE %s")
                params.append(f'%{model}%')
            
            if not include_evaluated:
                # 평가되지 않은 항목만 조회
                where_conditions.append("sqe.idx IS NULL")
            
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # 전체 개수 조회
            count_query = f"""
            SELECT COUNT(*) as total
            FROM result_logs rl
            LEFT JOIN synthesis_quality_evaluations sqe ON rl.idx = sqe.result_log_idx
            {where_clause}
            """
            cursor.execute(count_query, tuple(params))
            total = cursor.fetchone()['total']
            print(f"[Manual Evaluation] 전체 데이터 개수: {total}")
            
            # 이미지 목록 조회
            select_query = f"""
            SELECT 
                rl.idx,
                rl.person_url,
                rl.dress_url,
                rl.result_url,
                rl.model,
                rl.created_at,
                sqe.idx as evaluation_idx,
                sqe.is_success as evaluated_success
            FROM result_logs rl
            LEFT JOIN synthesis_quality_evaluations sqe ON rl.idx = sqe.result_log_idx
            {where_clause}
            ORDER BY sqe.idx IS NULL DESC, rl.created_at DESC
            """
            
            if limit is not None:
                # limit이 지정된 경우
                select_query += " LIMIT %s OFFSET %s"
                cursor.execute(select_query, tuple(params) + (limit, offset))
            else:
                # limit이 None인 경우 (전체 조회)
                # offset이 0이면 LIMIT 없이 전체 조회, offset이 있으면 LIMIT과 함께 사용
                if offset > 0:
                    # offset이 있으면 LIMIT을 매우 큰 값으로 설정
                    select_query += " LIMIT 18446744073709551615 OFFSET %s"
                    cursor.execute(select_query, tuple(params) + (offset,))
                else:
                    # offset이 0이면 LIMIT 절 없이 전체 조회
                    cursor.execute(select_query, tuple(params))
            results = cursor.fetchall()
            print(f"[Manual Evaluation] 실제 조회된 데이터 개수: {len(results)}")
            
            images = []
            for row in results:
                images.append({
                    "idx": row['idx'],
                    "person_url": row['person_url'] or "",
                    "dress_url": row['dress_url'] or "",
                    "result_url": row['result_url'] or "",
                    "model": row['model'] or "",
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "is_evaluated": row['evaluation_idx'] is not None,
                    "evaluated_success": bool(row['evaluated_success']) if row['evaluated_success'] is not None else None
                })
            
            return {
                "success": True,
                "images": images,
                "total": total
            }
    except Exception as e:
        print(f"[Manual Evaluation] 이미지 목록 조회 오류: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "images": [],
            "total": 0
        }
    finally:
        connection.close()


def save_manual_evaluation(
    result_log_idx: int,
    is_success: bool
) -> Dict:
    """
    수동 평가 결과를 저장합니다.
    
    Args:
        result_log_idx: result_logs.idx
        is_success: 성공 여부 (True=예, False=아니오)
    
    Returns:
        dict: {
            "success": bool,
            "evaluation_idx": Optional[int],
            "error": Optional[str]
        }
    """
    connection = get_db_connection()
    if connection is None:
        return {
            "success": False,
            "error": "DB 연결 실패"
        }
    
    try:
        with connection.cursor() as cursor:
            # 기존 평가 결과가 있는지 확인
            check_query = "SELECT idx FROM synthesis_quality_evaluations WHERE result_log_idx = %s"
            cursor.execute(check_query, (result_log_idx,))
            existing = cursor.fetchone()
            
            if existing:
                # 업데이트 (수동 평가는 quality_score를 -1로 설정하여 수동 평가임을 표시)
                update_query = """
                UPDATE synthesis_quality_evaluations
                SET quality_score = -1, 
                    quality_comment = '수동 평가',
                    is_success = %s, 
                    evaluated_at = CURRENT_TIMESTAMP
                WHERE result_log_idx = %s
                """
                cursor.execute(update_query, (is_success, result_log_idx))
                connection.commit()
                print(f"[Manual Evaluation] 평가 결과 업데이트 완료: result_log_idx={result_log_idx}, is_success={is_success}")
                return {
                    "success": True,
                    "evaluation_idx": existing['idx']
                }
            else:
                # 새로 삽입 (수동 평가는 quality_score를 -1로 설정하여 수동 평가임을 표시)
                insert_query = """
                INSERT INTO synthesis_quality_evaluations (result_log_idx, quality_score, quality_comment, is_success)
                VALUES (%s, -1, '수동 평가', %s)
                """
                cursor.execute(insert_query, (result_log_idx, is_success))
                connection.commit()
                evaluation_idx = cursor.lastrowid
                print(f"[Manual Evaluation] 평가 결과 저장 완료: idx={evaluation_idx}, result_log_idx={result_log_idx}, is_success={is_success}")
                return {
                    "success": True,
                    "evaluation_idx": evaluation_idx
                }
    except Exception as e:
        print(f"[Manual Evaluation] 평가 결과 저장 오류: {e}")
        traceback.print_exc()
        connection.rollback()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        connection.close()


def get_manual_evaluation_statistics(
    model: Optional[str] = None
) -> Dict:
    """
    수동 평가 통계를 조회합니다 (파이프라인별 성공률).
    
    Args:
        model: 모델 필터 (None이면 모든 모델)
    
    Returns:
        dict: {
            "success": bool,
            "overall": {
                "total": int,
                "success": int,
                "failed": int,
                "success_rate": float
            },
            "by_model": List[Dict]
        }
    """
    connection = get_db_connection()
    if connection is None:
        return {
            "success": False,
            "error": "DB 연결 실패"
        }
    
    try:
        with connection.cursor() as cursor:
            # 기본 WHERE 조건 (수동 평가만 - quality_comment가 '수동 평가'인 것)
            where_conditions = ["sqe.quality_comment = '수동 평가'"]
            params = []
            
            if model:
                where_conditions.append("rl.model LIKE %s")
                params.append(f'%{model}%')
            
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # 전체 통계
            overall_query = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN sqe.is_success = 1 THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN sqe.is_success = 0 THEN 1 ELSE 0 END) as failed
            FROM synthesis_quality_evaluations sqe
            INNER JOIN result_logs rl ON sqe.result_log_idx = rl.idx
            {where_clause}
            """
            cursor.execute(overall_query, tuple(params))
            overall_result = cursor.fetchone()
            
            total = int(overall_result['total'] or 0)
            success = int(overall_result['success'] or 0)
            failed = int(overall_result['failed'] or 0)
            success_rate = float((success / total * 100) if total > 0 else 0.0)
            
            # 모델별 통계
            by_model_query = f"""
            SELECT 
                rl.model,
                COUNT(*) as total,
                SUM(CASE WHEN sqe.is_success = 1 THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN sqe.is_success = 0 THEN 1 ELSE 0 END) as failed
            FROM synthesis_quality_evaluations sqe
            INNER JOIN result_logs rl ON sqe.result_log_idx = rl.idx
            {where_clause}
            GROUP BY rl.model
            ORDER BY total DESC
            """
            cursor.execute(by_model_query, tuple(params))
            by_model_results = cursor.fetchall()
            
            by_model = []
            for row in by_model_results:
                model_total = int(row['total'] or 0)
                model_success = int(row['success'] or 0)
                model_failed = int(row['failed'] or 0)
                model_success_rate = float((model_success / model_total * 100) if model_total > 0 else 0.0)
                
                by_model.append({
                    "model": row['model'],
                    "total": model_total,
                    "success": model_success,
                    "failed": model_failed,
                    "success_rate": round(model_success_rate, 2)
                })
            
            return {
                "success": True,
                "overall": {
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "success_rate": round(success_rate, 2)
                },
                "by_model": by_model
            }
    except Exception as e:
        print(f"[Manual Evaluation] 통계 조회 오류: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        connection.close()

