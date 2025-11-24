"""리뷰 라우터"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from services.database import get_db_connection
from schemas.review import ReviewCreate, ReviewResponse

router = APIRouter()


@router.post("/api/reviews", tags=["리뷰"], response_model=dict)
async def create_review(review: ReviewCreate):
    """
    리뷰 제출
    
    사용자가 서비스에 대한 리뷰를 제출합니다.
    
    Args:
        review: 리뷰 생성 데이터 (rating, content, category)
    
    Returns:
        dict: 성공 여부 및 메시지
    """
    # 카테고리 유효성 검사
    valid_categories = ['general', 'custom', 'analysis']
    if review.category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 카테고리입니다. 가능한 값: {', '.join(valid_categories)}"
        )
    
    connection = get_db_connection()
    if not connection:
        raise HTTPException(
            status_code=500,
            detail="데이터베이스 연결에 실패했습니다."
        )
    
    try:
        with connection.cursor() as cursor:
            insert_query = """
            INSERT INTO reviews (rating, content, category)
            VALUES (%s, %s, %s)
            """
            cursor.execute(
                insert_query,
                (
                    review.rating,
                    review.content if review.content else None,
                    review.category
                )
            )
            connection.commit()
            
            # 삽입된 리뷰 ID 가져오기
            review_idx = cursor.lastrowid
            
            return JSONResponse({
                "success": True,
                "message": "리뷰가 성공적으로 제출되었습니다.",
                "review_id": review_idx
            })
    except Exception as e:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"리뷰 제출 중 오류가 발생했습니다: {str(e)}"
        )
    finally:
        connection.close()


@router.get("/api/reviews", tags=["리뷰"], response_model=dict)
async def get_reviews(
    category: str = None,
    limit: int = 100,
    offset: int = 0
):
    """
    리뷰 목록 조회
    
    저장된 리뷰 목록을 조회합니다.
    
    Args:
        category: 카테고리 필터 (선택사항)
        limit: 조회할 최대 개수 (기본값: 100)
        offset: 건너뛸 개수 (기본값: 0)
    
    Returns:
        dict: 리뷰 목록 및 통계
    """
    connection = get_db_connection()
    if not connection:
        raise HTTPException(
            status_code=500,
            detail="데이터베이스 연결에 실패했습니다."
        )
    
    try:
        with connection.cursor() as cursor:
            # 카테고리 필터 적용
            if category:
                valid_categories = ['general', 'custom', 'analysis']
                if category not in valid_categories:
                    raise HTTPException(
                        status_code=400,
                        detail=f"유효하지 않은 카테고리입니다. 가능한 값: {', '.join(valid_categories)}"
                    )
                count_query = "SELECT COUNT(*) as total FROM reviews WHERE category = %s"
                select_query = """
                SELECT idx, rating, content, category, created_at
                FROM reviews
                WHERE category = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """
                cursor.execute(count_query, (category,))
                total = cursor.fetchone()['total']
                cursor.execute(select_query, (category, limit, offset))
            else:
                count_query = "SELECT COUNT(*) as total FROM reviews"
                select_query = """
                SELECT idx, rating, content, category, created_at
                FROM reviews
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """
                cursor.execute(count_query)
                total = cursor.fetchone()['total']
                cursor.execute(select_query, (limit, offset))
            
            reviews = cursor.fetchall()
            
            # 통계 정보 (리뷰가 없을 때도 처리)
            avg_rating = None
            if total > 0:
                cursor.execute("SELECT AVG(rating) as avg_rating FROM reviews")
                avg_result = cursor.fetchone()['avg_rating']
                if avg_result is not None:
                    try:
                        avg_rating = round(float(avg_result), 2)
                    except (ValueError, TypeError):
                        avg_rating = None
            
            # datetime 객체를 문자열로 변환
            formatted_reviews = []
            for review in reviews:
                formatted_review = dict(review)
                if 'created_at' in formatted_review and formatted_review['created_at']:
                    if hasattr(formatted_review['created_at'], 'isoformat'):
                        formatted_review['created_at'] = formatted_review['created_at'].isoformat()
                    elif isinstance(formatted_review['created_at'], str):
                        pass  # 이미 문자열인 경우
                formatted_reviews.append(formatted_review)
            
            return JSONResponse({
                "success": True,
                "total": total,
                "limit": limit,
                "offset": offset,
                "average_rating": avg_rating,
                "reviews": formatted_reviews
            })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"리뷰 조회 오류: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=f"리뷰 조회 중 오류가 발생했습니다: {str(e)}"
        )
    finally:
        if connection:
            connection.close()

