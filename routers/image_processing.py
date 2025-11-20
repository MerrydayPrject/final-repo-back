"""이미지 처리 라우터"""
import os
import io
import base64
import traceback
import numpy as np
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn

# 사용되지 않는 get_segformer_b2_processor와 get_segformer_b2_model 임포트 제거
from core.model_loader import (
    load_realesrgan_model, get_sdxl_pipeline, get_rtmpose_model
)
from core.xai_client import generate_image_from_text
from config.settings import GEMINI_FLASH_MODEL
from services.log_service import save_test_log
from core.s3_client import upload_log_to_s3
import time

router = APIRouter()


@router.post("/api/upscale", tags=["해상도 향상"])
async def upscale_image(
    file: UploadFile = File(..., description="업스케일할 이미지 파일"),
    scale: int = Form(4, description="업스케일 배율 (2, 4)")
):
    """
    Real-ESRGAN 해상도 향상
    
    이미지의 해상도와 질감을 향상시킵니다.
    """
    try:
        import cv2
        
        # Real-ESRGAN 모델 lazy loading
        realesrgan_model = load_realesrgan_model()
        
        if realesrgan_model is None:
            try:
                from realesrgan import RealESRGANer
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
                # Real-ESRGAN 모델 로드
                model_path = f'weights/RealESRGAN_x{scale}plus.pth'
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                       num_conv=32, upscale=scale, act_type='prelu')
                realesrgan_model = RealESRGANer(scale=scale, model_path=model_path, 
                                               model=model, tile=0, tile_pad=10, 
                                               pre_pad=0, half=False, device=device)
                print("Real-ESRGAN 모델 로딩 완료!")
            except Exception as e:
                # 모델 파일이 없으면 간단한 업스케일링 사용
                print(f"Real-ESRGAN 모델 로딩 실패, 간단한 업스케일링 사용: {e}")
                realesrgan_model = None
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        if realesrgan_model is not None:
            # Real-ESRGAN 사용
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            output, _ = realesrgan_model.enhance(img_bgr, outscale=scale)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(output_rgb)
        else:
            # 간단한 업스케일링 (Lanczos 리샘플링)
            new_size = (image.size[0] * scale, image.size[1] * scale)
            result_img = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_img.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "scale": scale,
            "original_size": image.size,
            "result_size": result_img.size,
            "message": f"해상도 향상 완료 ({scale}x 업스케일)"
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/color-harmonize", tags=["색상 보정"])
async def color_harmonize(
    file: UploadFile = File(..., description="색상 보정할 이미지 파일"),
    reference_file: Optional[UploadFile] = File(None, description="참조 이미지 (선택사항)")
):
    """
    Color Harmonization - 조명 및 색상 보정
    
    이미지의 조명과 색상을 조정하여 자연스러운 결과를 만듭니다.
    """
    try:
        import cv2
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # OpenCV 형식으로 변환
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        if reference_file:
            # 참조 이미지가 있으면 색상 전이
            ref_contents = await reference_file.read()
            ref_image = Image.open(io.BytesIO(ref_contents)).convert("RGB")
            ref_array = np.array(ref_image)
            ref_bgr = cv2.cvtColor(ref_array, cv2.COLOR_RGB2BGR)
            
            # LAB 색공간으로 변환
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
            
            # 색상 전이 (LAB 색공간에서)
            img_lab[:, :, 1] = ref_lab[:, :, 1]  # a 채널
            img_lab[:, :, 2] = ref_lab[:, :, 2]  # b 채널
            
            # BGR로 변환
            result_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        else:
            # 자동 색상 보정 (CLAHE 사용)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE 적용 (대비 향상)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # LAB 합성
            lab = cv2.merge([l, a, b])
            result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 결과 이미지 변환
        result_img = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_img.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "Color Harmonization 색상 보정 완료"
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/generate-shoes", tags=["구두 생성"])
async def generate_shoes(
    prompt: str = Form(..., description="구두 생성 프롬프트"),
    model_type: str = Form("gemini", description="사용할 모델 (gemini 또는 sdxl)")
):
    """
    구두 이미지 생성 (SDXL-LoRA 또는 Gemini 2.5 Image)
    
    프롬프트를 기반으로 구두 이미지를 생성합니다.
    """
    try:
        if model_type == "gemini":
            # Gemini 2.5 Image 사용
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return JSONResponse({
                    "success": False,
                    "error": "API key not found",
                    "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
                }, status_code=500)
            
            from google import genai
            client = genai.Client(api_key=api_key)
            
            # 텍스트 프롬프트로 이미지 생성
            response = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=[f"Generate a high-quality image of {prompt}. The image should be photorealistic and detailed."]
            )
            
            # 응답에서 이미지 추출
            image_parts = [
                part.inline_data.data
                for part in response.candidates[0].content.parts
                if hasattr(part, 'inline_data') and part.inline_data
            ]
            
            if image_parts:
                result_image_base64 = base64.b64encode(image_parts[0]).decode()
                return JSONResponse({
                    "success": True,
                    "result_image": f"data:image/png;base64,{result_image_base64}",
                    "model": GEMINI_FLASH_MODEL,
                    "message": "Gemini로 구두 이미지 생성 완료"
                })
            else:
                return JSONResponse({
                    "success": False,
                    "error": "No image generated",
                    "message": "Gemini API가 이미지를 생성하지 못했습니다."
                }, status_code=500)
        
        else:
            # SDXL-LoRA 사용
            sdxl_pipeline = get_sdxl_pipeline()
            
            if sdxl_pipeline is None:
                try:
                    from diffusers import StableDiffusionXLPipeline
                    
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32
                    )
                    sdxl_pipeline = sdxl_pipeline.to(device)
                    print("SDXL 파이프라인 로딩 완료!")
                except Exception as e:
                    return JSONResponse({
                        "success": False,
                        "error": f"모델 로딩 실패: {str(e)}",
                        "message": "SDXL 모델을 로드할 수 없습니다."
                    }, status_code=500)
            
            # 이미지 생성
            image = sdxl_pipeline(prompt=prompt, num_inference_steps=50).images[0]
            
            # 결과 이미지를 base64로 인코딩
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            result_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return JSONResponse({
                "success": True,
                "result_image": f"data:image/png;base64,{result_base64}",
                "model": "sdxl-base-1.0",
                "message": "SDXL로 구두 이미지 생성 완료"
            })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/generate-image-xai", tags=["이미지 생성"])
async def generate_image_xai(
    prompt: str = Form(..., description="이미지 생성 프롬프트"),
    model: Optional[str] = Form(None, description="사용할 모델 ID (선택사항, 기본값: grok-2-image)"),
    n: int = Form(1, description="생성할 이미지 수"),
    person_image: Optional[UploadFile] = File(None, description="사람 이미지 파일 (로깅용)"),
    dress_image: Optional[UploadFile] = File(None, description="드레스 이미지 파일 (로깅용)"),
    prompt_llm: Optional[str] = Form(None, description="프롬프트 생성에 사용된 LLM 모델")
):
    """
    x.ai API를 사용한 텍스트 to 이미지 생성
    
    프롬프트를 기반으로 x.ai API를 사용하여 이미지를 생성합니다.
    person_image와 dress_image가 제공되면 로그를 저장합니다.
    
    Note: 
    - x.ai API는 size 파라미터를 지원하지 않습니다.
    - 모델이 지정되지 않으면 기본값 "grok-2-image"를 사용합니다.
    """
    start_time = time.time()
    
    try:
        # 모델이 지정되지 않으면 기본값 사용 (xai_client에서 처리)
        result = generate_image_from_text(
            prompt=prompt,
            model=model,
            n=n
        )
        
        run_time = time.time() - start_time
        
        # 모델명 결정 (prompt_llm이 있으면 그것을 포함)
        model_name = result.get("model", "x.ai-default")
        if prompt_llm:
            model_name = f"{prompt_llm}+{model_name}"
        
        if result["success"]:
            # 로깅을 위한 이미지 처리
            person_s3_url = ""
            dress_s3_url = None
            result_s3_url = ""
            
            # 결과 이미지를 S3에 업로드
            if result.get("result_image"):
                try:
                    # base64 이미지를 디코딩
                    if result["result_image"].startswith("data:image"):
                        base64_data = result["result_image"].split(",")[1]
                    else:
                        base64_data = result["result_image"]
                    
                    image_bytes = base64.b64decode(base64_data)
                    result_s3_url = upload_log_to_s3(image_bytes, model_name, "result") or ""
                except Exception as e:
                    print(f"결과 이미지 S3 업로드 실패: {e}")
            
            # 사람 이미지와 드레스 이미지가 제공된 경우 S3에 업로드
            if person_image:
                try:
                    person_bytes = await person_image.read()
                    person_s3_url = upload_log_to_s3(person_bytes, model_name, "person") or ""
                except Exception as e:
                    print(f"사람 이미지 S3 업로드 실패: {e}")
            
            if dress_image:
                try:
                    dress_bytes = await dress_image.read()
                    dress_s3_url = upload_log_to_s3(dress_bytes, model_name, "dress") or None
                except Exception as e:
                    print(f"드레스 이미지 S3 업로드 실패: {e}")
            
            # 로그 저장
            save_test_log(
                person_url=person_s3_url,
                dress_url=dress_s3_url,
                result_url=result_s3_url,
                model=model_name,
                prompt=prompt,
                success=True,
                run_time=run_time
            )
            
            return JSONResponse({
                "success": True,
                "result_image": result["result_image"],
                "model": model_name,
                "message": result.get("message", "x.ai로 이미지 생성 완료")
            })
        else:
            # 실패 시에도 로그 저장 (이미지가 있는 경우)
            if person_image or dress_image:
                person_s3_url = ""
                dress_s3_url = None
                
                if person_image:
                    try:
                        person_bytes = await person_image.read()
                        person_s3_url = upload_log_to_s3(person_bytes, model_name, "person") or ""
                    except Exception as e:
                        print(f"사람 이미지 S3 업로드 실패: {e}")
                
                if dress_image:
                    try:
                        dress_bytes = await dress_image.read()
                        dress_s3_url = upload_log_to_s3(dress_bytes, model_name, "dress") or None
                    except Exception as e:
                        print(f"드레스 이미지 S3 업로드 실패: {e}")
                
                save_test_log(
                    person_url=person_s3_url,
                    dress_url=dress_s3_url,
                    result_url="",
                    model=model_name,
                    prompt=prompt,
                    success=False,
                    run_time=run_time
                )
            
            return JSONResponse({
                "success": False,
                "error": result.get("error", "Unknown error"),
                "message": result.get("message", "이미지 생성 실패")
            }, status_code=500)
        
    except Exception as e:
        traceback.print_exc()
        run_time = time.time() - start_time
        
        # 예외 발생 시에도 로그 저장 시도
        if person_image or dress_image:
            model_name = model or "x.ai-default"
            if prompt_llm:
                model_name = f"{prompt_llm}+{model_name}"
            
            person_s3_url = ""
            dress_s3_url = None
            
            if person_image:
                try:
                    person_bytes = await person_image.read()
                    person_s3_url = upload_log_to_s3(person_bytes, model_name, "person") or ""
                except:
                    pass
            
            if dress_image:
                try:
                    dress_bytes = await dress_image.read()
                    dress_s3_url = upload_log_to_s3(dress_bytes, model_name, "dress") or None
                except:
                    pass
            
            save_test_log(
                person_url=person_s3_url,
                dress_url=dress_s3_url,
                result_url="",
                model=model_name,
                prompt=prompt,
                success=False,
                run_time=run_time
            )
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/tps-warp", tags=["TPS Warp"])
async def tps_warp(
    shoes_image: UploadFile = File(..., description="구두 이미지 파일"),
    person_image: UploadFile = File(..., description="사람 이미지 파일")
):
    """
    TPS Warp - 구두 워핑 및 착용 삽입
    
    구두 이미지를 사람 이미지의 발 위치에 맞게 워핑하여 합성합니다.
    """
    try:
        import cv2
        
        # 이미지 읽기
        shoes_contents = await shoes_image.read()
        person_contents = await person_image.read()
        
        shoes_img = Image.open(io.BytesIO(shoes_contents)).convert("RGB")
        person_img = Image.open(io.BytesIO(person_contents)).convert("RGB")
        
        # 원본 이미지들을 base64로 인코딩
        shoes_buffered = io.BytesIO()
        shoes_img.save(shoes_buffered, format="PNG")
        shoes_base64 = base64.b64encode(shoes_buffered.getvalue()).decode()
        
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        # OpenCV 형식으로 변환
        shoes_cv = cv2.cvtColor(np.array(shoes_img), cv2.COLOR_RGB2BGR)
        person_cv = cv2.cvtColor(np.array(person_img), cv2.COLOR_RGB2BGR)
        
        # 간단한 TPS Warp 구현
        # 실제로는 발 위치를 감지하고 정교한 워핑 필요
        h, w = person_cv.shape[:2]
        
        # 구두 이미지 리사이즈
        shoes_resized = cv2.resize(shoes_cv, (w // 4, h // 4))
        
        # 사람 이미지의 하단에 구두 합성 (간단한 버전)
        result_cv = person_cv.copy()
        y_offset = h - shoes_resized.shape[0] - 50
        x_offset = w // 2 - shoes_resized.shape[1] // 2
        
        # ROI 추출
        roi = result_cv[y_offset:y_offset+shoes_resized.shape[0], 
                       x_offset:x_offset+shoes_resized.shape[1]]
        
        # 알파 블렌딩
        mask = np.ones(shoes_resized.shape, dtype=shoes_resized.dtype) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_inv = cv2.bitwise_not(mask)
        
        # 배경과 전경 분리
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img_fg = cv2.bitwise_and(shoes_resized, shoes_resized, mask=mask)
        
        # 합성
        dst = cv2.add(img_bg, img_fg)
        result_cv[y_offset:y_offset+shoes_resized.shape[0], 
                 x_offset:x_offset+shoes_resized.shape[1]] = dst
        
        # 결과 이미지 변환
        result_img = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        
        # 결과 이미지를 base64로 인코딩
        result_buffered = io.BytesIO()
        result_img.save(result_buffered, format="PNG")
        result_base64 = base64.b64encode(result_buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "shoes_image": f"data:image/png;base64,{shoes_base64}",
            "person_image": f"data:image/png;base64,{person_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "message": "TPS Warp 구두 합성 완료 (참고: 정교한 워핑 알고리즘 필요)"
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)


@router.post("/api/pose-estimation", tags=["포즈 인식"])
async def pose_estimation(file: UploadFile = File(..., description="포즈 인식할 이미지 파일")):
    """
    RTMPose-s 포즈/관절 키포인트 인식
    
    인체의 포즈와 관절 키포인트를 인식하여 위치 정보를 반환합니다.
    """
    try:
        import cv2
        import mmcv
        
        # RTMPose 모델 lazy loading
        rtmpose_model = get_rtmpose_model()
        
        if rtmpose_model is None:
            try:
                from mmpose.apis import init_model, inference_top_down_pose_model
                
                config_file = 'configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py'
                checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
                
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                rtmpose_model = init_model(config_file, checkpoint_file, device=device)
                print("RTMPose-s 모델 로딩 완료!")
            except Exception as e:
                return JSONResponse({
                    "success": False,
                    "error": f"모델 로딩 실패: {str(e)}",
                    "message": "RTMPose-s 모델을 로드할 수 없습니다. mmpose 설치 및 설정을 확인하세요."
                }, status_code=500)
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # mmcv 형식으로 변환
        img_array = np.array(image)
        img_bgr = mmcv.imconvert(img_array, 'RGB', 'BGR')
        
        # 포즈 추론
        from mmpose.apis import inference_top_down_pose_model
        pose_results, _ = inference_top_down_pose_model(rtmpose_model, img_bgr)
        
        # 키포인트를 이미지에 시각화
        from mmpose.visualization import draw_skeleton_and_kp
        
        vis_img = draw_skeleton_and_kp(
            img_array,
            pose_results,
            kp_thr=0.3,
            skeleton_style='mmpose'
        )
        
        # 결과 이미지를 base64로 인코딩
        vis_pil = Image.fromarray(vis_img)
        buffered_result = io.BytesIO()
        vis_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 키포인트 정보 추출
        keypoints = []
        if pose_results and len(pose_results) > 0:
            for person in pose_results:
                if 'keypoints' in person:
                    keypoints.append(person['keypoints'].tolist())
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "keypoints": keypoints,
            "num_persons": len(keypoints),
            "message": f"{len(keypoints)}명의 포즈 감지됨"
        })
        
    except ImportError as e:
        return JSONResponse({
            "success": False,
            "error": "mmpose 라이브러리 미설치",
            "message": "mmpose를 설치하세요: pip install mmpose>=0.31.0"
        }, status_code=500)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)