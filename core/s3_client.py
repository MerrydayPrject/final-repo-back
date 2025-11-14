"""S3 클라이언트"""
import os
import time
import boto3
from botocore.exceptions import ClientError
from typing import Optional


def upload_to_s3(file_content: bytes, file_name: str, content_type: str = "image/png", folder: str = "dresses") -> Optional[str]:
    """
    S3에 파일 업로드
    
    Args:
        file_content: 파일 내용 (bytes)
        file_name: 파일명
        content_type: MIME 타입
        folder: S3 폴더 경로 (기본값: "dresses")
    
    Returns:
        S3 URL 또는 None (실패 시)
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # S3에 업로드
        s3_key = f"{folder}/{file_name}"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        
        # S3 URL 생성
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        return s3_url
        
    except ClientError as e:
        print(f"S3 업로드 오류: {e}")
        return None
    except Exception as e:
        print(f"S3 업로드 중 예상치 못한 오류: {e}")
        return None


def upload_log_to_s3(file_content: bytes, model_id: str, image_type: str, content_type: str = "image/png") -> Optional[str]:
    """
    S3 logs 폴더에 테스트 이미지 업로드 (별도 S3 계정/버킷 사용)
    
    Args:
        file_content: 파일 내용 (bytes)
        model_id: 모델 ID
        image_type: 이미지 타입 (person, dress, result)
        content_type: MIME 타입
    
    Returns:
        S3 URL 또는 None (실패 시)
    """
    try:
        # 별도 S3 계정 환경변수 사용
        aws_access_key = os.getenv("LOGS_AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("LOGS_AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("LOGS_AWS_S3_BUCKET_NAME")
        region = os.getenv("LOGS_AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("로그용 S3 설정이 완료되지 않았습니다. (LOGS_AWS_*)")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # 타임스탬프 기반 파일명 생성
        timestamp = int(time.time() * 1000)
        file_name = f"{timestamp}_{model_id}_{image_type}.png"
        s3_key = f"logs/{file_name}"
        
        # S3에 업로드
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        
        # S3 URL 생성
        s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        return s3_url
        
    except ClientError as e:
        print(f"로그용 S3 업로드 오류: {e}")
        return None
    except Exception as e:
        print(f"로그용 S3 업로드 중 예상치 못한 오류: {e}")
        return None


def delete_from_s3(file_name: str) -> bool:
    """
    S3에서 파일 삭제
    
    Args:
        file_name: 삭제할 파일명
    
    Returns:
        삭제 성공 여부 (True/False)
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return False
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # S3 키 생성 (업로드 시와 동일한 형식)
        s3_key = f"dresses/{file_name}"
        
        # S3에서 삭제
        s3_client.delete_object(
            Bucket=bucket_name,
            Key=s3_key
        )
        
        print(f"S3에서 이미지 삭제 완료: {s3_key}")
        return True
        
    except ClientError as e:
        print(f"S3 삭제 오류: {e}")
        return False
    except Exception as e:
        print(f"S3 삭제 중 예상치 못한 오류: {e}")
        return False


def get_s3_image(file_name: str) -> Optional[bytes]:
    """
    S3에서 이미지 다운로드
    
    Args:
        file_name: 파일명 (예: "Adress1.JPG")
    
    Returns:
        이미지 바이트 데이터 또는 None (실패 시)
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # S3에서 파일 다운로드
        s3_key = f"dresses/{file_name}"
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"S3에 파일이 없습니다: {s3_key}")
            else:
                print(f"S3 다운로드 오류: {e}")
            return None
    except Exception as e:
        print(f"S3 이미지 다운로드 중 예상치 못한 오류: {e}")
        return None

