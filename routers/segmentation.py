"""세그멘테이션 라우터"""
from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
import io
import base64
import numpy as np
from PIL import Image
# import torch  # 주석 처리: torch/transformers 미사용
# import torch.nn as nn  # 주석 처리: torch/transformers 미사용

# from core.model_loader import get_processor, get_model  # 주석 처리: torch/transformers 미사용
from schemas.segmentation import SegmentationResponse, ErrorResponse, LabelInfo
from config.settings import LABELS

router = APIRouter()


# ============================================================
# 의상 합성 고품화 기능 주석 처리 (torch/transformers 미사용)
# ============================================================
# @router.post("/api/segment", tags=["세그멘테이션"])
# async def segment_dress(file: UploadFile = File(..., description="세그멘테이션할 이미지 파일")):
#     """
#     드레스 세그멘테이션 (웨딩드레스 누끼)
#     
#     업로드된 이미지에서 드레스(레이블 7)를 감지하고 배경을 제거합니다.
#     """
#     try:
#         processor = get_processor()
#         model = get_model()
#         
#         if processor is None or model is None:
#             return JSONResponse({
#                 "success": False,
#                 "error": "Model not loaded",
#                 "message": "모델이 로드되지 않았습니다. 서버를 재시작해주세요."
#             }, status_code=503)
#         
#         # 이미지 읽기
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         original_size = image.size
#         
#         # 원본 이미지를 base64로 인코딩
#         buffered_original = io.BytesIO()
#         image.save(buffered_original, format="PNG")
#         original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
#         
#         # 모델 추론
#         inputs = processor(images=image, return_tensors="pt")
#         
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits.cpu()
#         
#         # 업샘플링
#         upsampled_logits = nn.functional.interpolate(
#             logits,
#             size=original_size[::-1],  # (height, width)
#             mode="bilinear",
#             align_corners=False,
#         )
#         
#         # 세그멘테이션 마스크 생성
#         pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
#         
#         # 드레스 마스크 생성 (레이블 7: Dress)
#         dress_mask = (pred_seg == 7).astype(np.uint8) * 255
#         
#         # 원본 이미지를 numpy 배열로 변환
#         image_array = np.array(image)
#         
#         # 누끼 이미지 생성 (RGBA)
#         result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
#         result_image[:, :, :3] = image_array  # RGB 채널
#         result_image[:, :, 3] = dress_mask    # 알파 채널
#         
#         # PIL 이미지로 변환
#         result_pil = Image.fromarray(result_image, mode='RGBA')
#         
#         # 결과 이미지를 base64로 인코딩
#         buffered_result = io.BytesIO()
#         result_pil.save(buffered_result, format="PNG")
#         result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
#         
#         # 드레스가 감지되었는지 확인
#         dress_pixels = int(np.sum(pred_seg == 7))
#         total_pixels = int(pred_seg.size)
#         dress_percentage = float((dress_pixels / total_pixels) * 100)
#         
#         return JSONResponse({
#             "success": True,
#             "original_image": f"data:image/png;base64,{original_base64}",
#             "result_image": f"data:image/png;base64,{result_base64}",
#             "dress_detected": bool(dress_pixels > 0),
#             "dress_percentage": round(dress_percentage, 2),
#             "message": f"드레스 영역: {dress_percentage:.2f}% 감지됨" if dress_pixels > 0 else "드레스가 감지되지 않았습니다."
#         })
#         
#     except Exception as e:
#         return JSONResponse({
#             "success": False,
#             "error": str(e),
#             "message": f"처리 중 오류 발생: {str(e)}"
#         }, status_code=500)


# @router.post("/api/segment-custom", tags=["세그멘테이션"])
# async def segment_custom(
#     file: UploadFile = File(..., description="세그멘테이션할 이미지 파일"),
#     labels: str = Query(..., description="추출할 레이블 ID (쉼표로 구분, 예: 4,5,6,7)")
# ):
#     """
#     커스텀 레이블 세그멘테이션
#     
#     지정한 레이블들만 추출하여 배경을 제거합니다.
#     """
#     try:
#         processor = get_processor()
#         model = get_model()
#         
#         if processor is None or model is None:
#             return JSONResponse({
#                 "success": False,
#                 "error": "Model not loaded",
#                 "message": "모델이 로드되지 않았습니다. 서버를 재시작해주세요."
#             }, status_code=503)
#         
#         # 레이블 파싱
#         label_ids = [int(l.strip()) for l in labels.split(",")]
#         
#         # 이미지 읽기
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         original_size = image.size
#         
#         # 원본 이미지를 base64로 인코딩
#         buffered_original = io.BytesIO()
#         image.save(buffered_original, format="PNG")
#         original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
#         
#         # 모델 추론
#         inputs = processor(images=image, return_tensors="pt")
#         
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits.cpu()
#         
#         # 업샘플링
#         upsampled_logits = nn.functional.interpolate(
#             logits,
#             size=original_size[::-1],
#             mode="bilinear",
#             align_corners=False,
#         )
#         
#         # 세그멘테이션 마스크 생성
#         pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
#         
#         # 선택한 레이블들의 마스크 생성
#         combined_mask = np.zeros_like(pred_seg, dtype=bool)
#         for label_id in label_ids:
#             combined_mask |= (pred_seg == label_id)
#         
#         mask = combined_mask.astype(np.uint8) * 255
#         
#         # 원본 이미지를 numpy 배열로 변환
#         image_array = np.array(image)
#         
#         # 누끼 이미지 생성 (RGBA)
#         result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
#         result_image[:, :, :3] = image_array
#         result_image[:, :, 3] = mask
#         
#         # PIL 이미지로 변환
#         result_pil = Image.fromarray(result_image, mode='RGBA')
#         
#         # 결과 이미지를 base64로 인코딩
#         buffered_result = io.BytesIO()
#         result_pil.save(buffered_result, format="PNG")
#         result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
#         
#         # 각 레이블의 픽셀 수 계산
#         detected_labels = []
#         total_pixels = int(pred_seg.size)
#         for label_id in label_ids:
#             pixels = int(np.sum(pred_seg == label_id))
#             if pixels > 0:
#                 detected_labels.append(LabelInfo(
#                     id=label_id,
#                     name=LABELS.get(label_id, "Unknown"),
#                     percentage=round((pixels / total_pixels) * 100, 2)
#                 ))
#         
#         total_detected = int(np.sum(combined_mask))
#         
#         return JSONResponse({
#             "success": True,
#             "original_image": f"data:image/png;base64,{original_base64}",
#             "result_image": f"data:image/png;base64,{result_base64}",
#             "requested_labels": [{"id": lid, "name": LABELS.get(lid, "Unknown")} for lid in label_ids],
#             "detected_labels": [label.dict() for label in detected_labels],
#             "total_percentage": round((total_detected / total_pixels) * 100, 2),
#             "message": f"{len(detected_labels)}개의 레이블 감지됨" if detected_labels else "선택한 레이블이 감지되지 않았습니다."
#         })
#         
#     except Exception as e:
#         return JSONResponse({
#             "success": False,
#             "error": str(e),
#             "message": f"처리 중 오류 발생: {str(e)}"
#         }, status_code=500)


# @router.post("/api/segment-b0", tags=["세그멘테이션"])
# async def segment_b0(file: UploadFile = File(..., description="세그멘테이션할 이미지 파일")):
#     """
#     SegFormer B0 세그멘테이션 (배경 제거/옷 영역 인식)
#     
#     matei-dorian/segformer-b0-finetuned-human-parsing 모델을 사용하여
#     이미지에서 배경을 제거하고 옷 영역을 인식합니다.
#     """
#     from core.model_loader import get_segformer_b2_processor, get_segformer_b2_model
#     from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
#     
#     try:
#         # 모델 lazy loading
#         segformer_b0_processor = get_segformer_b2_processor()
#         segformer_b0_model = get_segformer_b2_model()
#         
#         if segformer_b0_processor is None or segformer_b0_model is None:
#             try:
#                 segformer_b0_processor = SegformerImageProcessor.from_pretrained("matei-dorian/segformer-b0-finetuned-human-parsing")
#                 segformer_b0_model = AutoModelForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b0-finetuned-human-parsing")
#                 segformer_b0_model.eval()
#                 # 전역 변수에 저장 (core/model_loader.py에서 관리)
#                 from core.model_loader import segformer_b2_processor, segformer_b2_model
#                 import core.model_loader as ml
#                 ml.segformer_b2_processor = segformer_b0_processor
#                 ml.segformer_b2_model = segformer_b0_model
#                 print("SegFormer B0 모델 로딩 완료!")
#             except Exception as e:
#                 return JSONResponse({
#                     "success": False,
#                     "error": f"모델 로딩 실패: {str(e)}",
#                     "message": "SegFormer B0 모델을 로드할 수 없습니다."
#                 }, status_code=500)
#         
#         # 이미지 읽기
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         original_size = image.size
#         
#         # 원본 이미지를 base64로 인코딩
#         buffered_original = io.BytesIO()
#         image.save(buffered_original, format="PNG")
#         original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
#         
#         # 모델 추론
#         inputs = segformer_b0_processor(images=image, return_tensors="pt")
#         
#         with torch.no_grad():
#             outputs = segformer_b0_model(**inputs)
#             logits = outputs.logits.cpu()
#         
#         # 업샘플링
#         upsampled_logits = nn.functional.interpolate(
#             logits,
#             size=original_size[::-1],
#             mode="bilinear",
#             align_corners=False,
#         )
#         
#         # 세그멘테이션 마스크 생성 (배경이 아닌 모든 것)
#         pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
#         mask = (pred_seg != 0).astype(np.uint8) * 255
#         
#         # 원본 이미지를 numpy 배열로 변환
#         image_array = np.array(image)
#         
#         # 누끼 이미지 생성 (RGBA)
#         result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
#         result_image[:, :, :3] = image_array
#         result_image[:, :, 3] = mask
#         
#         # PIL 이미지로 변환
#         result_pil = Image.fromarray(result_image, mode='RGBA')
#         
#         # 결과 이미지를 base64로 인코딩
#         buffered_result = io.BytesIO()
#         result_pil.save(buffered_result, format="PNG")
#         result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
#         
#         # 전경 픽셀 비율 계산
#         foreground_pixels = int(np.sum(pred_seg != 0))
#         total_pixels = int(pred_seg.size)
#         foreground_percentage = round((foreground_pixels / total_pixels) * 100, 2)
#         
#         return JSONResponse({
#             "success": True,
#             "original_image": f"data:image/png;base64,{original_base64}",
#             "result_image": f"data:image/png;base64,{result_base64}",
#             "foreground_percentage": foreground_percentage,
#             "message": f"SegFormer B0 세그멘테이션 완료 (전경 영역: {foreground_percentage}%)"
#         })
#         
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return JSONResponse({
#             "success": False,
#             "error": str(e),
#             "message": f"처리 중 오류 발생: {str(e)}"
#         }, status_code=500)


# @router.post("/api/remove-background", tags=["세그멘테이션"])
# async def remove_background(file: UploadFile = File(..., description="배경을 제거할 이미지 파일")):
#     """
#     전체 배경 제거 (인물만 추출)
#     
#     배경(레이블 0)을 제거하고 인물과 의류만 남깁니다.
#     """
#     try:
#         processor = get_processor()
#         model = get_model()
#         
#         if processor is None or model is None:
#             return JSONResponse({
#                 "success": False,
#                 "error": "Model not loaded",
#                 "message": "모델이 로드되지 않았습니다. 서버를 재시작해주세요."
#             }, status_code=503)
#         
#         # 이미지 읽기
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         original_size = image.size
#         
#         # 원본 이미지를 base64로 인코딩
#         buffered_original = io.BytesIO()
#         image.save(buffered_original, format="PNG")
#         original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
#         
#         # 모델 추론
#         inputs = processor(images=image, return_tensors="pt")
#         
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits.cpu()
#         
#         # 업샘플링
#         upsampled_logits = nn.functional.interpolate(
#             logits,
#             size=original_size[::-1],
#             mode="bilinear",
#             align_corners=False,
#         )
#         
#         # 세그멘테이션 마스크 생성
#         pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
#         
#         # 배경이 아닌 모든 것을 포함하는 마스크
#         mask = (pred_seg != 0).astype(np.uint8) * 255
#         
#         # 원본 이미지를 numpy 배열로 변환
#         image_array = np.array(image)
#         
#         # 누끼 이미지 생성 (RGBA)
#         result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
#         result_image[:, :, :3] = image_array
#         result_image[:, :, 3] = mask
#         
#         # PIL 이미지로 변환
#         result_pil = Image.fromarray(result_image, mode='RGBA')
#         
#         # 결과 이미지를 base64로 인코딩
#         buffered_result = io.BytesIO()
#         result_pil.save(buffered_result, format="PNG")
#         result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
#         
#         # 배경이 아닌 픽셀 수 계산
#         foreground_pixels = int(np.sum(pred_seg != 0))
#         total_pixels = int(pred_seg.size)
#         foreground_percentage = round((foreground_pixels / total_pixels) * 100, 2)
#         
#         return JSONResponse({
#             "success": True,
#             "original_image": f"data:image/png;base64,{original_base64}",
#             "result_image": f"data:image/png;base64,{result_base64}",
#             "foreground_percentage": foreground_percentage,
#             "message": f"배경 제거 완료 (인물 영역: {foreground_percentage}%)"
#         })
#         
#     except Exception as e:
#         return JSONResponse({
#             "success": False,
#             "error": str(e),
#             "message": f"처리 중 오류 발생: {str(e)}"
#         }, status_code=500)


# @router.post("/api/analyze", tags=["분석"])
# async def analyze_image(file: UploadFile = File(..., description="분석할 이미지 파일")):
#     """
#     이미지 전체 분석
#     
#     이미지에서 모든 레이블을 감지하고 각 레이블의 비율을 분석합니다.
#     누끼 처리 없이 분석 정보만 반환합니다.
#     """
#     try:
#         processor = get_processor()
#         model = get_model()
#         
#         if processor is None or model is None:
#             return JSONResponse({
#                 "success": False,
#                 "error": "Model not loaded",
#                 "message": "모델이 로드되지 않았습니다. 서버를 재시작해주세요."
#             }, status_code=503)
#         
#         # 이미지 읽기
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         original_size = image.size
#         
#         # 모델 추론
#         inputs = processor(images=image, return_tensors="pt")
#         
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits.cpu()
#         
#         # 업샘플링
#         upsampled_logits = nn.functional.interpolate(
#             logits,
#             size=original_size[::-1],
#             mode="bilinear",
#             align_corners=False,
#         )
#         
#         # 세그멘테이션 마스크 생성
#         pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
#         
#         # 각 레이블의 픽셀 수 계산
#         total_pixels = int(pred_seg.size)
#         detected_labels = []
#         
#         for label_id, label_name in LABELS.items():
#             pixels = int(np.sum(pred_seg == label_id))
#             percentage = round((pixels / total_pixels) * 100, 2)
#             if pixels > 0:
#                 detected_labels.append(LabelInfo(
#                     id=label_id,
#                     name=label_name,
#                     percentage=percentage
#                 ))
#         
#         # 비율 순으로 정렬
#         detected_labels.sort(key=lambda x: x.percentage, reverse=True)
#         
#         return JSONResponse({
#             "success": True,
#             "image_size": {"width": original_size[0], "height": original_size[1]},
#             "total_pixels": total_pixels,
#             "detected_labels": [label.dict() for label in detected_labels],
#             "total_detected": len(detected_labels),
#             "message": f"총 {len(detected_labels)}개의 레이블 감지됨"
#         })
#         
#     except Exception as e:
#         return JSONResponse({
#             "success": False,
#             "error": str(e),
#             "message": f"처리 중 오류 발생: {str(e)}"
#         }, status_code=500)

