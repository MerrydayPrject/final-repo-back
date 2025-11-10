"""
3D 이미지 변환 API
드레스 이미지를 3D로 변환하는 기능을 제공합니다.
"""

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import numpy as np
import io
import base64
import time
from pathlib import Path
import cv2

# FastAPI 앱 초기화
app = FastAPI(
    title="3D 이미지 변환 API",
    description="드레스 이미지를 3D로 변환하는 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

# 디렉토리 생성
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# 템플릿 및 정적 파일 설정
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 전역 변수
depth_model = None
device = None

# ===================== 모델 로딩 =====================

def load_depth_model():
    """Depth Estimation 모델 로딩"""
    global depth_model, device
    
    if depth_model is not None:
        return depth_model
    
    try:
        print("Depth 모델 로딩 중...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")
        
        # MiDaS 모델 로드 (경량 버전)
        model_type = "DPT_Large"  # 또는 "MiDaS_small" for faster inference
        
        try:
            depth_model = torch.hub.load("intel-isl/MiDaS", model_type)
            depth_model.to(device)
            depth_model.eval()
            print("MiDaS 모델 로딩 완료!")
        except Exception as e:
            print(f"MiDaS 로딩 실패: {e}")
            print("간단한 depth 추정으로 대체합니다.")
            depth_model = "simple"
        
        return depth_model
        
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None

# ===================== 3D 변환 함수 =====================

def estimate_depth_simple(image_array):
    """
    간단한 Depth 추정 (모델 없이)
    Sobel 엣지 검출과 블러를 이용한 기본적인 depth map 생성
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Sobel 엣지 검출
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # 정규화
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min()) * 255
    sobel = sobel.astype(np.uint8)
    
    # 가우시안 블러로 부드럽게
    depth = cv2.GaussianBlur(sobel, (21, 21), 0)
    
    # 반전 (밝은 부분이 가까운 것으로)
    depth = 255 - depth
    
    return depth

def estimate_depth_midas(image_array, model):
    """MiDaS 모델을 사용한 Depth 추정"""
    try:
        # 이미지 전처리
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if "DPT" in str(model):
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        # RGB to BGR (MiDaS expects BGR)
        img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        input_batch = transform(img_bgr).to(device)
        
        # 추론
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_array.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # numpy로 변환
        depth = prediction.cpu().numpy()
        
        # 정규화 (0-255)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype(np.uint8)
        
        return depth
        
    except Exception as e:
        print(f"MiDaS Depth 추정 실패: {e}")
        return estimate_depth_simple(image_array)

def create_normal_map(depth_map):
    """Depth Map에서 Normal Map 생성"""
    # Sobel을 사용하여 gradients 계산
    depth_float = depth_map.astype(np.float32)
    
    sobelx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # Normal vector 계산
    # Z는 항상 양수 (표면이 카메라를 향함)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2 + 1)
    
    normal_x = -sobelx / sobel_magnitude
    normal_y = -sobely / sobely_magnitude
    normal_z = 1.0 / sobel_magnitude
    
    # -1~1 범위를 0~255로 변환
    normal_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((normal_x + 1) * 0.5 * 255).astype(np.uint8)  # R
    normal_map[:, :, 1] = ((normal_y + 1) * 0.5 * 255).astype(np.uint8)  # G
    normal_map[:, :, 2] = ((normal_z) * 255).astype(np.uint8)  # B
    
    return normal_map

def create_3d_effect(image_array, depth_map, strength=0.05):
    """
    Depth Map을 사용하여 3D 효과 생성
    양안 시차를 시뮬레이션하여 입체감 생성
    """
    h, w = depth_map.shape
    
    # Depth를 displacement로 변환
    displacement = (depth_map.astype(np.float32) / 255.0) * strength * w
    
    # 좌안 이미지 생성 (왼쪽으로 shift)
    left_image = np.zeros_like(image_array)
    for y in range(h):
        for x in range(w):
            shift = int(displacement[y, x])
            src_x = max(0, min(w - 1, x - shift))
            left_image[y, x] = image_array[y, src_x]
    
    # 우안 이미지 생성 (오른쪽으로 shift)
    right_image = np.zeros_like(image_array)
    for y in range(h):
        for x in range(w):
            shift = int(displacement[y, x])
            src_x = max(0, min(w - 1, x + shift))
            right_image[y, x] = image_array[y, src_x]
    
    return left_image, right_image

# ===================== API 엔드포인트 =====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩"""
    print("=" * 50)
    print("3D 이미지 변환 서버 시작")
    print("=" * 50)
    load_depth_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("3d_conversion_test.html", {"request": request})

@app.post("/api/convert-to-3d")
async def convert_to_3d(
    image: UploadFile = File(..., description="변환할 이미지"),
    mode: str = Form("depth", description="변환 모드: depth, normal, stereo")
):
    """
    이미지를 3D로 변환
    
    - mode: depth (Depth Map), normal (Normal Map), stereo (입체 효과)
    """
    start_time = time.time()
    
    try:
        # 이미지 읽기
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # RGB 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Depth 모델 로드
        model = load_depth_model()
        
        # Depth Map 생성
        if model == "simple" or model is None:
            depth_map = estimate_depth_simple(img_array)
        else:
            depth_map = estimate_depth_midas(img_array, model)
        
        result = {
            "success": True,
            "processing_time": round(time.time() - start_time, 2),
            "image_size": f"{img_array.shape[1]}x{img_array.shape[0]}"
        }
        
        # Depth Map을 이미지로 변환
        depth_img = Image.fromarray(depth_map)
        depth_buffer = io.BytesIO()
        depth_img.save(depth_buffer, format='PNG')
        result["depth_map"] = base64.b64encode(depth_buffer.getvalue()).decode('utf-8')
        
        # Normal Map 생성
        if mode in ["normal", "all"]:
            normal_map = create_normal_map(depth_map)
            normal_img = Image.fromarray(normal_map)
            normal_buffer = io.BytesIO()
            normal_img.save(normal_buffer, format='PNG')
            result["normal_map"] = base64.b64encode(normal_buffer.getvalue()).decode('utf-8')
        
        # 입체 효과 생성
        if mode in ["stereo", "all"]:
            left_img, right_img = create_3d_effect(img_array, depth_map)
            
            left_pil = Image.fromarray(left_img)
            left_buffer = io.BytesIO()
            left_pil.save(left_buffer, format='PNG')
            result["left_image"] = base64.b64encode(left_buffer.getvalue()).decode('utf-8')
            
            right_pil = Image.fromarray(right_img)
            right_buffer = io.BytesIO()
            right_pil.save(right_buffer, format='PNG')
            result["right_image"] = base64.b64encode(right_buffer.getvalue()).decode('utf-8')
        
        return JSONResponse(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "processing_time": round(time.time() - start_time, 2)
        }, status_code=500)

# ===================== 서버 실행 =====================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("3D 이미지 변환 테스트 서버")
    print("URL: http://localhost:8003")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8003)

