"""S3 클라이언트"""
import os
import time
import boto3
from botocore.exceptions import ClientError
from typing import Optional, List
from datetime import timedelta


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


def check_file_exists_in_s3(file_name: str, folder: str = "dresses") -> bool:
    """
    S3에 파일이 존재하는지 확인
    
    Args:
        file_name: 파일명
        folder: S3 폴더 경로 (기본값: "dresses")
    
    Returns:
        파일 존재 여부 (True/False)
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
        
        s3_key = f"{folder}/{file_name}"
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                print(f"S3 파일 존재 확인 오류: {e}")
                return False
    except Exception as e:
        print(f"S3 파일 존재 확인 중 예상치 못한 오류: {e}")
        return False


def list_files_in_s3_folder(folder: str = "dresses", max_keys: int = 1000) -> List[dict]:
    """
    S3 폴더에 있는 파일 목록 조회
    
    Args:
        folder: S3 폴더 경로 (기본값: "dresses")
        max_keys: 최대 조회 개수 (기본값: 1000)
    
    Returns:
        파일 정보 리스트 [{"key": "folder/file.jpg", "size": 12345, "last_modified": "..."}, ...]
    """
    try:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("AWS S3 설정이 완료되지 않았습니다.")
            return []
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        files = []
        prefix = f"{folder}/"
        
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        
                        # 폴더 자체는 제외 (마지막이 /로 끝나는 경우)
                        if s3_key.endswith('/'):
                            continue
                        
                        # prefix 제거하여 상대 경로 추출
                        if s3_key.startswith(prefix):
                            file_name = s3_key[len(prefix):]
                        else:
                            file_name = s3_key
                        
                        if file_name:  # 빈 문자열이 아닌 경우만
                            # Pre-signed URL 생성 (1시간 유효)
                            try:
                                presigned_url = s3_client.generate_presigned_url(
                                    'get_object',
                                    Params={'Bucket': bucket_name, 'Key': s3_key},
                                    ExpiresIn=3600  # 1시간
                                )
                            except Exception as e:
                                print(f"[WARN] Pre-signed URL 생성 실패: {s3_key}, 오류: {e}")
                                # 실패 시 일반 URL 사용
                                presigned_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
                            
                            files.append({
                                "key": s3_key,
                                "file_name": file_name,  # 상대 경로 포함 (예: "A라인/image.jpg")
                                "size": obj['Size'],
                                "last_modified": obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                                "url": presigned_url
                            })
            
            return files
        except ClientError as e:
            print(f"S3 파일 목록 조회 오류: {e}")
            return []
    except Exception as e:
        print(f"S3 파일 목록 조회 중 예상치 못한 오류: {e}")
        return []


def get_logs_s3_image(file_name: str) -> Optional[bytes]:
    """
    로그용 S3에서 이미지 다운로드 (별도 S3 계정/버킷 사용)
    
    Args:
        file_name: 파일명 (예: "1763098638885_gemini-compose_result.png")
    
    Returns:
        이미지 바이트 데이터 또는 None (실패 시)
    """
    try:
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
        
        # S3에서 파일 다운로드 (logs 폴더)
        s3_key = f"logs/{file_name}"
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"로그용 S3에 파일이 없습니다: {s3_key}")
            else:
                print(f"로그용 S3 다운로드 오류: {e}")
            return None
    except Exception as e:
        print(f"로그용 S3 이미지 다운로드 중 예상치 못한 오류: {e}")
        return None
