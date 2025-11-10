"""
이미지 보정 서버 (포트 8003)
Stable Diffusion Inpainting + GFPGAN 기반 이미지 보정
"""
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import io
import base64
import torch
import cv2
import numpy as np
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)
from controlnet_aux import OpenposeDetector
from pathlib import Path
import os
import sys
from typing import Optional, List, Dict, Tuple
import time

# MediaPipe BodyAnalysisService import (상대 경로)
sys.path.append(str(Path(__file__).parent.parent))
try:
    from body_analysis_test.body_analysis import BodyAnalysisService
except ImportError:
    print("⚠️  BodyAnalysisService를 import할 수 없습니다. MediaPipe 포즈 추출 기능이 제한될 수 있습니다.")
    BodyAnalysisService = None

# FastAPI 앱 초기화
app = FastAPI(
    title="이미지 보정 API",
    description="GFPGAN 얼굴 보정 + Inpainting 신체 보정 서비스",
    version="3.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "http://localhost:8002", "http://localhost:8003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 설정
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

# 전역 변수로 모델 저장
controlnet_pipe = None
openpose_detector_model = None
body_analysis_service = None


def normalize_instruction(text: Optional[str]) -> str:
    """자주 발생하는 오타와 띄어쓰기를 정규화"""
    if not text:
        return ""
    normalized = text.strip()
    replacements = {
        "어꺠": "어깨",
        "어깨을": "어깨를",
        "허리을": "허리를",
        "엉덩이을": "엉덩이를",
        "좁혀": "좁게",
        "좁혀줘": "좁게 해줘",
        "좁혀주세요": "좁게 해주세요",
        "좁혀달라": "좁게 해달라",
    }
    for wrong, correct in replacements.items():
        normalized = normalized.replace(wrong, correct)
    return normalized

CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "diffusers/controlnet-openpose-sdxl-1.0")
SDXL_INPAINT_MODEL_ID = os.getenv("SDXL_INPAINT_MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpaint")
OPENPOSE_DETECTOR_ID = os.getenv("OPENPOSE_DETECTOR_ID", "lllyasviel/ControlNet")


def load_models():
    """ControlNet + SDXL Inpaint + MediaPipe 로드"""
    global controlnet_pipe, openpose_detector_model, body_analysis_service

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if openpose_detector_model is None:
        try:
            print("[1/3] OpenPose Detector 로딩 중...")
            openpose_detector_model = OpenposeDetector.from_pretrained(OPENPOSE_DETECTOR_ID)
            print("✅ OpenPose Detector 로드 완료")
        except Exception as e:
            print(f"❌ OpenPose Detector 로드 실패: {e}")
            openpose_detector_model = None

    if controlnet_pipe is None:
        try:
            print("[2/3] ControlNet + SDXL Inpaint 로딩 중...")
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL_ID,
                torch_dtype=dtype,
                use_safetensors=True,
            )

            pipe_kwargs = {
                "torch_dtype": dtype,
                "controlnet": controlnet,
                "use_safetensors": True,
            }
            if device.type == "cuda":
                pipe_kwargs["variant"] = "fp16"

            controlnet_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                SDXL_INPAINT_MODEL_ID,
                **pipe_kwargs,
            )

            if device.type == "cuda":
                controlnet_pipe.to(device)
                try:
                    controlnet_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    print("⚠️  xFormers 메모리 최적화 활성화 실패 (무시)")
            else:
                controlnet_pipe.to(device)
                try:
                    controlnet_pipe.enable_model_cpu_offload()
                except Exception:
                    print("⚠️  CPU 오프로딩 활성화 실패 (무시)")

            controlnet_pipe.set_progress_bar_config(disable=True)
            print("✅ ControlNet + SDXL Inpaint 로드 완료")
        except Exception as e:
            print(f"❌ ControlNet + SDXL Inpaint 로드 실패: {e}")
            controlnet_pipe = None

    # 3. BodyAnalysisService (MediaPipe)
    if body_analysis_service is None and BodyAnalysisService is not None:
        print("[3/3] MediaPipe BodyAnalysisService 로딩 중...")
        try:
            model_path = Path(__file__).parent.parent / "body_analysis_test" / "models" / "pose_landmarker_lite.task"
            body_analysis_service = BodyAnalysisService(model_path=str(model_path) if model_path.exists() else None)
            if body_analysis_service.is_initialized:
                print("✅ MediaPipe BodyAnalysisService 로드 완료")
            else:
                print("⚠️  MediaPipe BodyAnalysisService 초기화 실패")
                body_analysis_service = None
        except Exception as e:
            print(f"❌ MediaPipe BodyAnalysisService 로드 실패: {e}")
            body_analysis_service = None

def create_face_protection_mask(landmarks: List[Dict], image_size: Tuple[int, int]) -> Image.Image:
    """
    얼굴 영역 보호 마스크 생성 (얼굴 영역을 검은색으로 표시하여 보호)
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        image_size: 이미지 크기 (width, height)
    
    Returns:
        마스크 이미지 (검은색=얼굴 영역 보호, 흰색=편집 가능)
    """
    mask = Image.new("L", image_size, 255)  # 전체 흰색 (편집 가능)
    draw = ImageDraw.Draw(mask)
    
    if not landmarks or len(landmarks) < 33:
        return mask
    
    # MediaPipe 랜드마크 인덱스 (얼굴)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 7
    RIGHT_EAR = 8
    
    width, height = image_size
    
    # 얼굴 영역 추정 (코, 눈, 귀 기준)
    if len(landmarks) > max(NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR):
        nose_x = int(landmarks[NOSE]["x"] * width)
        nose_y = int(landmarks[NOSE]["y"] * height)
        
        # 얼굴 영역 크기 추정 (더 크게 보호)
        face_width = width * 0.35  # 얼굴 폭 추정 (25% → 35%)
        face_height = height * 0.4  # 얼굴 높이 추정 (30% → 40%)
        
        # 얼굴 영역을 검은색으로 (보호) - 더 넓은 영역
        x0 = max(0, int(nose_x - face_width / 2))
        x1 = min(width - 1, int(nose_x + face_width / 2))
        y0 = max(0, int(nose_y - face_height * 0.4))  # 위쪽 더 많이 보호
        y1 = min(height - 1, int(nose_y + face_height * 0.6))  # 아래쪽도 더 보호
        
        if x1 > x0 and y1 > y0:
            draw.ellipse(
                [(x0, y0), (x1, y1)],
                fill=0  # 검은색 (얼굴 보호)
            )
    
    return mask

def create_mask_for_region(
    landmarks: List[Dict],
    region: str,
    image_size: Tuple[int, int],
    protect_face: bool = True,
    mask_scale: float = 1.0,
) -> Image.Image:
    """
    특정 영역(어깨, 허리, 엉덩이)에 대한 마스크 생성 (얼굴 보호 옵션)
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트 (인덱스 기반)
        region: 영역 이름 ("shoulder", "waist", "hip")
        image_size: 이미지 크기 (width, height)
        protect_face: 얼굴 영역 보호 여부
    
    Returns:
        마스크 이미지 (흰색=편집 영역, 검은색=유지 영역)
    """
    mask = Image.new("L", image_size, 0)  # 검은색 (유지)
    draw = ImageDraw.Draw(mask)
    
    if not landmarks or len(landmarks) < 33:
        return mask
    
    # MediaPipe 랜드마크 인덱스
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    
    width, height = image_size
    
    mask_scale = max(0.7, min(mask_scale, 1.8))  # 안정적인 범위 제한

    if region == "shoulder" and len(landmarks) > max(LEFT_SHOULDER, RIGHT_SHOULDER):
        # 어깨 영역 마스크
        left_shoulder_x = int(landmarks[LEFT_SHOULDER]["x"] * width)
        right_shoulder_x = int(landmarks[RIGHT_SHOULDER]["x"] * width)
        y = int((landmarks[LEFT_SHOULDER]["y"] + landmarks[RIGHT_SHOULDER]["y"]) / 2 * height)
        
        # 좌표 정렬 (left_x < right_x 보장)
        min_x = min(left_shoulder_x, right_shoulder_x)
        max_x = max(left_shoulder_x, right_shoulder_x)
        
        padding = int(30 * mask_scale)
        vertical_padding = int(40 * mask_scale)
        x0 = max(0, min_x - padding)  # 이미지 범위 내로 제한
        x1 = min(width - 1, max_x + padding)
        y0 = max(0, y - vertical_padding)
        y1 = min(height - 1, y + vertical_padding)
        
        # 유효한 좌표인지 확인
        if x1 > x0 and y1 > y0:
            draw.ellipse(
                [(x0, y0), (x1, y1)],
                fill=255
            )
    
    elif region == "waist" and len(landmarks) > max(LEFT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        # 허리 영역 마스크 (어깨와 엉덩이 사이)
        shoulder_y = int(landmarks[LEFT_SHOULDER]["y"] * height)
        hip_y = int(landmarks[LEFT_HIP]["y"] * height)
        waist_y = (shoulder_y + hip_y) // 2
        
        left_hip_x = int(landmarks[LEFT_HIP]["x"] * width)
        right_hip_x = int(landmarks[RIGHT_HIP]["x"] * width)
        
        # 좌표 정렬 (left_x < right_x 보장)
        min_x = min(left_hip_x, right_hip_x)
        max_x = max(left_hip_x, right_hip_x)
        
        padding_x = int(25 * mask_scale)
        padding_y = int(50 * mask_scale)
        x0 = max(0, min_x - padding_x)
        x1 = min(width - 1, max_x + padding_x)
        y0 = max(0, waist_y - padding_y)
        y1 = min(height - 1, waist_y + padding_y)
        
        # 유효한 좌표인지 확인
        if x1 > x0 and y1 > y0:
            draw.ellipse(
                [(x0, y0), (x1, y1)],
                fill=255
            )
    
    elif region == "hip" and len(landmarks) > max(LEFT_HIP, RIGHT_HIP):
        # 엉덩이 영역 마스크
        left_hip_x = int(landmarks[LEFT_HIP]["x"] * width)
        right_hip_x = int(landmarks[RIGHT_HIP]["x"] * width)
        y = int((landmarks[LEFT_HIP]["y"] + landmarks[RIGHT_HIP]["y"]) / 2 * height)
        
        # 좌표 정렬 (left_x < right_x 보장)
        min_x = min(left_hip_x, right_hip_x)
        max_x = max(left_hip_x, right_hip_x)
        
        padding = int(45 * mask_scale)
        vertical_padding = int(60 * mask_scale)
        x0 = max(0, min_x - padding)
        x1 = min(width - 1, max_x + padding)
        y0 = max(0, y - vertical_padding)
        y1 = min(height - 1, y + vertical_padding)
        
        # 유효한 좌표인지 확인
        if x1 > x0 and y1 > y0:
            draw.ellipse(
                [(x0, y0), (x1, y1)],
                fill=255
            )
    
    # 얼굴 영역 보호 (얼굴 영역을 검은색으로 덮어서 편집 방지)
    if protect_face:
        face_mask = create_face_protection_mask(landmarks, image_size)
        # 얼굴 영역을 마스크에서 제외 (검은색으로 덮음)
        mask_array = np.array(mask)
        face_mask_array = np.array(face_mask)
        # 얼굴 영역(검은색)은 편집하지 않도록 마스크에서 제외
        mask_array[face_mask_array == 0] = 0  # 얼굴 영역은 검은색(유지)
        mask = Image.fromarray(mask_array)
    
    return mask


def adjust_pose_for_instruction(
    pose_image: Image.Image,
    landmarks: Optional[List[Dict]],
    instruction: str,
) -> Image.Image:
    """지시문에 맞춰 포즈 이미지(어깨/허리/엉덩이) 보조 선을 추가"""
    if pose_image is None or not landmarks:
        return pose_image

    edited_pose = pose_image.copy()
    draw = ImageDraw.Draw(edited_pose)
    width, height = edited_pose.size

    def to_xy(index: int) -> Tuple[float, float]:
        if len(landmarks) <= index:
            return (0.0, 0.0)
        return (
            float(landmarks[index]["x"] * width),
            float(landmarks[index]["y"] * height),
        )

    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24

    if "어깨" in instruction and len(landmarks) > max(LEFT_SHOULDER, RIGHT_SHOULDER):
        left = to_xy(LEFT_SHOULDER)
        right = to_xy(RIGHT_SHOULDER)
        span = max(1.0, abs(right[0] - left[0]))
        offset = span * 0.18
        if "좁" in instruction:
            left_new = (left[0] + offset, left[1])
            right_new = (right[0] - offset, right[1])
        else:
            left_new = (left[0] - offset, left[1])
            right_new = (right[0] + offset, right[1])

        draw.line([left_new, right_new], fill=(0, 255, 0), width=12)
        for point in (left_new, right_new):
            draw.ellipse(
                (point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8),
                fill=(0, 255, 0),
            )

    if "허리" in instruction and len(landmarks) > max(LEFT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        shoulder_mid = (
            (to_xy(LEFT_SHOULDER)[0] + to_xy(RIGHT_SHOULDER)[0]) / 2,
            (to_xy(LEFT_SHOULDER)[1] + to_xy(RIGHT_SHOULDER)[1]) / 2,
        )
        hip_left = to_xy(LEFT_HIP)
        hip_right = to_xy(RIGHT_HIP)
        waist_y = (shoulder_mid[1] + hip_left[1]) / 2
        span = max(1.0, abs(hip_right[0] - hip_left[0]))
        offset = span * 0.15
        if any(keyword in instruction for keyword in ["얇", "작"]):
            left_waist = (hip_left[0] + offset, waist_y)
            right_waist = (hip_right[0] - offset, waist_y)
        else:
            left_waist = (hip_left[0] - offset, waist_y)
            right_waist = (hip_right[0] + offset, waist_y)

        draw.line([left_waist, right_waist], fill=(0, 200, 255), width=10)

    if "엉덩" in instruction and len(landmarks) > max(LEFT_HIP, RIGHT_HIP):
        hip_left = to_xy(LEFT_HIP)
        hip_right = to_xy(RIGHT_HIP)
        span = max(1.0, abs(hip_right[0] - hip_left[0]))
        offset = span * 0.2
        if "작" in instruction:
            hip_left_new = (hip_left[0] + offset, hip_left[1])
            hip_right_new = (hip_right[0] - offset, hip_right[1])
        else:
            hip_left_new = (hip_left[0] - offset, hip_left[1])
            hip_right_new = (hip_right[0] + offset, hip_right[1])

        draw.line([hip_left_new, hip_right_new], fill=(255, 128, 0), width=14)
        for point in (hip_left_new, hip_right_new):
            draw.ellipse(
                (point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9),
                fill=(255, 128, 0),
            )

    return edited_pose


def build_control_image(
    image: Image.Image,
    instruction: str,
    landmarks: Optional[List[Dict]],
) -> Optional[Image.Image]:
    """ControlNet용 포즈 이미지 생성 및 지시문에 맞게 보정"""
    if openpose_detector_model is None:
        return None
    try:
        pose_image = openpose_detector_model(image)
        if pose_image is None:
            return None
        if landmarks:
            pose_image = adjust_pose_for_instruction(pose_image, landmarks, instruction)
        return pose_image
    except Exception as exc:
        print(f"⚠️  포즈 이미지 생성 실패: {exc}")
        return None


def parse_body_edit_params(instruction: str) -> Tuple[float, int, float, int, float]:
    """신체 보정 강도 파라미터 추출"""
    instruction_lower = instruction.lower()
    strength = 0.7
    steps = 34
    mask_scale = 1.2
    iterations = 2
    control_scale = 0.7

    strong_keywords = ["많이", "확", "대폭", "강하게", "크게", "significantly", "dramatic"]
    gentle_keywords = ["살짝", "조금", "약하게", "미세하게", "slightly", "subtle"]

    if any(keyword in instruction_lower for keyword in strong_keywords):
        strength = min(0.90, strength + 0.15)
        steps = min(40, steps + 6)
        mask_scale = min(1.6, mask_scale + 0.3)
        iterations = 3
        control_scale = min(1.0, control_scale + 0.25)
    elif any(keyword in instruction_lower for keyword in gentle_keywords):
        strength = max(0.45, strength - 0.15)
        steps = max(24, steps - 8)
        mask_scale = max(0.9, mask_scale - 0.2)
        iterations = 1
        control_scale = max(0.45, control_scale - 0.25)

    return strength, steps, mask_scale, iterations, control_scale


def translate_instruction(instruction: str) -> str:
    """한국어 요청을 영어 프롬프트로 변환"""
    instruction_lower = instruction.lower()
    prompt_parts = []
    
    # 형태 조작
    if "어깨" in instruction:
        if "넓" in instruction or "넓게" in instruction or "크게" in instruction:
            prompt_parts.append("make shoulders wider")
        elif "좁" in instruction or "좁게" in instruction or "작게" in instruction:
            prompt_parts.append("make shoulders narrower")
    
    if "허리" in instruction:
        if "얇" in instruction or "얇게" in instruction or "작게" in instruction:
            prompt_parts.append("make waist thinner")
        elif "넓" in instruction or "넓게" in instruction:
            prompt_parts.append("make waist wider")
    
    if "엉덩이" in instruction or "엉덩" in instruction:
        if "작" in instruction or "작게" in instruction:
            prompt_parts.append("make hips smaller")
        elif "넓" in instruction or "넓게" in instruction or "크게" in instruction:
            prompt_parts.append("make hips larger")
    
    # 배경 변경
    background_keywords = {
        "교회": "church",
        "해변": "beach",
        "바다": "ocean",
        "정원": "garden",
        "공원": "park",
        "스튜디오": "studio",
        "카페": "cafe",
        "호텔": "hotel",
        "웨딩홀": "wedding hall"
    }
    
    for kor, eng in background_keywords.items():
        if kor in instruction:
            prompt_parts.append(f"{eng} background")
            break
    
    if "배경" in instruction:
        if "블러" in instruction or "흐릿" in instruction:
            prompt_parts.append("blurred background")
        elif "흰색" in instruction or "하얀" in instruction:
            prompt_parts.append("white background")
    
    # 스타일 변경
    style_keywords = {
        "우아": "elegant",
        "모던": "modern",
        "캐주얼": "casual",
        "로맨틱": "romantic"
    }
    
    for keyword, eng in style_keywords.items():
        if keyword in instruction:
            prompt_parts.append(f"{eng} style")
            break
    
    # 분위기
    if "밝" in instruction:
        prompt_parts.append("bright")
    elif "어둡" in instruction:
        prompt_parts.append("dark")
    
    if not prompt_parts:
        prompt_parts.append("natural and realistic")
    
    return ", ".join(prompt_parts)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("=" * 60)
    print("이미지 보정 서버 시작 중...")
    print("ControlNet OpenPose + Stable Diffusion Inpainting")
    print("=" * 60)
    load_models()
    print("=" * 60)
    print("서버 준비 완료!")
    print("=" * 60)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """테스트 페이지"""
    if templates:
        return templates.TemplateResponse("enhancement_test.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>이미지 보정 서버</title></head>
            <body>
                <h1>이미지 보정 서버</h1>
                <p>API 문서: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "controlnet_inpaint": controlnet_pipe is not None,
        "openpose_detector": openpose_detector_model is not None,
        "body_analysis": body_analysis_service is not None and body_analysis_service.is_initialized if body_analysis_service else False
    }

@app.post("/api/enhance-image")
async def enhance_image(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    use_gfpgan: bool = Form(True),
    gfpgan_weight: float = Form(0.3)
):
    """
    이미지 보정 API (ControlNet OpenPose + Stable Diffusion XL Inpaint 기반)

    지원 기능:

    **신체 보정:**
    - 어깨: "어깨 좁게", "어깨 넓게"
    - 허리: "허리 얇게", "허리 넓게"
    - 엉덩이: "엉덩이 작게", "엉덩이 크게"

    **배경/스타일 변경:**
    - "해변 배경", "우아한 분위기" 등 translate_instruction에서 지원하는 키워드

    Args:
        file: 입력 이미지
        instruction: 보정 요청사항 (한국어, 선택사항)
        use_gfpgan: (Deprecated) 기존 UI 호환용. 현재 파이프라인에서는 무시됩니다.
        gfpgan_weight: (Deprecated) 기존 UI 호환용. 현재 파이프라인에서는 무시됩니다.
    """
    try:
        original_instruction = instruction or ""
        instruction = normalize_instruction(original_instruction)

        if controlnet_pipe is None or openpose_detector_model is None:
            load_models()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size

        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"이미지 리사이징: {image.size}")

        result = image
        processing_notes: List[str] = []
        last_prompt_used: Optional[str] = None

        needs_body_edit = any(
            keyword in instruction for keyword in ["어깨", "허리", "엉덩이", "키", "다리"]
        )

        if needs_body_edit and controlnet_pipe is not None:
            (
                body_strength,
                body_steps,
                mask_scale,
                body_iterations,
                control_scale,
            ) = parse_body_edit_params(instruction)
            print(
                "[1단계] 신체 보정 시작... "
                f"(strength={body_strength:.2f}, steps={body_steps}, mask_scale={mask_scale:.2f}, "
                f"iterations={body_iterations}, control_scale={control_scale:.2f})"
            )
            start_time = time.time()

            landmarks = None
            if body_analysis_service is not None:
                landmarks = body_analysis_service.extract_landmarks(image)

            control_image = build_control_image(result, instruction, landmarks)
            if control_image and control_image.size != result.size:
                control_image = control_image.resize(result.size, Image.Resampling.BILINEAR)

            face_original_array = None
            face_mask_bool = None
            if landmarks and len(landmarks) >= 33:
                face_mask = create_face_protection_mask(landmarks, result.size)
                face_mask_array = np.array(face_mask)
                face_mask_bool = face_mask_array == 0
                face_original_array = np.array(result).copy()
                print("✅ 얼굴 영역 추출 완료 (보호)")

            if landmarks and len(landmarks) >= 33:
                performed_edit = False

                def run_controlnet_edit(
                    current_image: Image.Image,
                    mask_image: Image.Image,
                    prompt_text: str,
                ) -> Optional[Image.Image]:
                    nonlocal control_image
                    if control_image is None:
                        control_image = build_control_image(current_image, instruction, landmarks)
                        if control_image and control_image.size != current_image.size:
                            control_image = control_image.resize(
                                current_image.size, Image.Resampling.BILINEAR
                            )
                    if control_image is None:
                        return None
                    return controlnet_pipe(
                        prompt=prompt_text,
                        image=current_image,
                        mask_image=mask_image,
                        control_image=control_image,
                        num_inference_steps=body_steps,
                        strength=body_strength,
                        guidance_scale=5.5,
                        negative_prompt="distorted face, deformed body, extra limbs, artifacts",
                        controlnet_conditioning_scale=control_scale,
                    ).images[0]

                if "어깨" in instruction:
                    mask = create_mask_for_region(
                        landmarks, "shoulder", result.size, protect_face=True, mask_scale=mask_scale
                    )
                    if "넓" in instruction or "넓게" in instruction or "크게" in instruction:
                        additional_prompt = ", increase shoulder width by 25 percent, make shoulders noticeably wider"
                    else:
                        additional_prompt = ", reduce shoulder width by 25 percent, make shoulders noticeably narrower"
                    prompt_inpaint = (
                        translate_instruction(instruction)
                        + additional_prompt
                        + ", keep face completely unchanged, keep original image style, natural, realistic, high quality, detailed, preserve face"
                    )
                    current_image = result
                    for _ in range(body_iterations):
                        current_result = run_controlnet_edit(current_image, mask, prompt_inpaint)
                        if current_result is None:
                            print("⚠️  ControlNet 포즈 이미지가 없어 어깨 편집을 건너뜁니다.")
                            processing_notes.append("controlnet_pose_missing")
                            break
                        current_image = current_result
                    result = current_image
                    last_prompt_used = prompt_inpaint

                    if face_mask_bool is not None and face_original_array is not None:
                        if result.size != image.size:
                            result = result.resize(image.size, Image.Resampling.LANCZOS)
                        result_array = np.array(result)
                        if result_array.shape[:2] == face_mask_bool.shape:
                            result_array[face_mask_bool] = face_original_array[face_mask_bool]
                            result = Image.fromarray(result_array)
                            print("✅ 얼굴 영역 원본 복원 완료")
                        else:
                            print(
                                f"⚠️  이미지 크기 불일치: result={result_array.shape}, mask={face_mask_bool.shape}, face={face_original_array.shape}"
                            )

                    print("✅ 어깨 영역 보정 완료")
                    performed_edit = True

                if "허리" in instruction:
                    mask = create_mask_for_region(
                        landmarks, "waist", result.size, protect_face=True, mask_scale=mask_scale
                    )
                    if "얇" in instruction or "얇게" in instruction or "작게" in instruction:
                        additional_prompt = ", emphasize a slimmer waistline, reduce waist circumference by 20 percent"
                    else:
                        additional_prompt = ", increase waist width by 20 percent, make waist noticeably wider"
                    prompt_inpaint = (
                        translate_instruction(instruction)
                        + additional_prompt
                        + ", keep face completely unchanged, keep original image style, natural, realistic, high quality, detailed, preserve face"
                    )
                    current_image = result
                    for _ in range(body_iterations):
                        current_result = run_controlnet_edit(current_image, mask, prompt_inpaint)
                        if current_result is None:
                            print("⚠️  ControlNet 포즈 이미지가 없어 허리 편집을 건너뜁니다.")
                            processing_notes.append("controlnet_pose_missing")
                            break
                        current_image = current_result
                    result = current_image
                    last_prompt_used = prompt_inpaint

                    if face_mask_bool is not None and face_original_array is not None:
                        if result.size != image.size:
                            result = result.resize(image.size, Image.Resampling.LANCZOS)
                        result_array = np.array(result)
                        if result_array.shape[:2] == face_mask_bool.shape:
                            result_array[face_mask_bool] = face_original_array[face_mask_bool]
                            result = Image.fromarray(result_array)
                            print("✅ 얼굴 영역 원본 복원 완료")
                        else:
                            print(
                                f"⚠️  이미지 크기 불일치: result={result_array.shape}, mask={face_mask_bool.shape}, face={face_original_array.shape}"
                            )

                    print("✅ 허리 영역 보정 완료")
                    performed_edit = True

                if "엉덩이" in instruction:
                    mask = create_mask_for_region(
                        landmarks, "hip", result.size, protect_face=True, mask_scale=mask_scale
                    )
                    if "작" in instruction or "작게" in instruction:
                        additional_prompt = ", reduce hip width by 20 percent, make hips noticeably smaller"
                    else:
                        additional_prompt = ", increase hip width by 20 percent, make hips noticeably larger"
                    prompt_inpaint = (
                        translate_instruction(instruction)
                        + additional_prompt
                        + ", keep face completely unchanged, keep original image style, natural, realistic, high quality, detailed, preserve face"
                    )
                    current_image = result
                    for _ in range(body_iterations):
                        current_result = run_controlnet_edit(current_image, mask, prompt_inpaint)
                        if current_result is None:
                            print("⚠️  ControlNet 포즈 이미지가 없어 엉덩이 편집을 건너뜁니다.")
                            processing_notes.append("controlnet_pose_missing")
                            break
                        current_image = current_result
                    result = current_image
                    last_prompt_used = prompt_inpaint

                    if face_mask_bool is not None and face_original_array is not None:
                        if result.size != image.size:
                            result = result.resize(image.size, Image.Resampling.LANCZOS)
                        result_array = np.array(result)
                        if result_array.shape[:2] == face_mask_bool.shape:
                            result_array[face_mask_bool] = face_original_array[face_mask_bool]
                            result = Image.fromarray(result_array)
                            print("✅ 얼굴 영역 원본 복원 완료")
                        else:
                            print(
                                f"⚠️  이미지 크기 불일치: result={result_array.shape}, mask={face_mask_bool.shape}, face={face_original_array.shape}"
                            )

                    print("✅ 엉덩이 영역 보정 완료")
                    performed_edit = True

                if not performed_edit:
                    result = image
            else:
                print("⚠️  MediaPipe 랜드마크 추출 실패로 신체 보정을 건너뜁니다.")
                processing_notes.append("pose_landmark_missing")

            elapsed_time = time.time() - start_time
            print(f"[1단계 완료] 신체 보정 처리 시간: {elapsed_time:.2f}초")

        face_keywords = ["주름", "피부", "톤", "얼굴", "하얗게", "밝게", "피부톤", "피부결", "기미", "잡티"]
        if any(keyword in instruction for keyword in face_keywords):
            processing_notes.append("face_refine_not_supported")

        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        message = "ControlNet 기반 이미지 보정이 완료되었습니다."
        if "face_refine_not_supported" in processing_notes:
            message += " (얼굴 톤/피부 세부 보정은 현재 파이프라인에서 별도 지원되지 않습니다.)"

        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "prompt_used": last_prompt_used,
            "notes": processing_notes,
            "message": message
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"이미지 보정 중 오류 발생: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    
    port = 8003
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"잘못된 포트 번호: {sys.argv[1]}. 기본 포트 8003 사용")
    
    print("=" * 60)
    print("이미지 보정 서버 시작...")
    print("ControlNet OpenPose + SDXL Inpaint")
    print(f"접속 주소: http://localhost:{port}")
    print(f"API 문서: http://localhost:{port}/docs")
    print(f"테스트 페이지: http://localhost:{port}/")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
