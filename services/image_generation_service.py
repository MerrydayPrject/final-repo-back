"""Stable Diffusion과 ControlNet, IP-Adapter를 사용한 이미지 생성 서비스"""
from PIL import Image
from core.model_loader import get_controlnet_pipeline
import torch

# IP-Adapter 로드 여부 추적
_ip_adapter_loaded = False

def _ensure_ip_adapter_loaded(pipe):
    """IP-Adapter가 로드되어 있는지 확인하고 없으면 로드합니다."""
    global _ip_adapter_loaded
    if not _ip_adapter_loaded:
        try:
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin"
            )
            _ip_adapter_loaded = True
        except Exception as e:
            print(f"IP-Adapter 로드 실패: {e}")
            _ip_adapter_loaded = False
    return _ip_adapter_loaded

async def generate_image_with_pose(prompt: str, pose_image: Image.Image, reference_image: Image.Image = None):
    """
    주어진 프롬프트와 포즈 이미지를 사용하여 이미지를 생성합니다.
    reference_image가 제공되면 IP-Adapter를 사용하여 얼굴/스타일을 보존합니다.
    """
    pipe = get_controlnet_pipeline()

    if not pipe:
        raise RuntimeError("이미지 생성 파이프라인을 로드할 수 없습니다.")

    # 입력 이미지 전처리
    pose_image = pose_image.convert("RGB").resize((512, 768))
    
    # 재현성을 위한 시드 설정
    generator = torch.manual_seed(0)

    # IP-Adapter 사용 여부 결정
    use_ip_adapter = reference_image is not None
    
    if use_ip_adapter:
        # IP-Adapter 로드
        ip_loaded = _ensure_ip_adapter_loaded(pipe)
        
        if ip_loaded:
            # reference 이미지 전처리
            reference_image = reference_image.convert("RGB").resize((512, 768))
            
            # IP-Adapter와 ControlNet을 함께 사용하여 생성
            pipe.set_ip_adapter_scale(0.6)  # 얼굴 보존 강도
            
            generated_image = pipe(
                prompt,
                num_inference_steps=30,
                generator=generator,
                image=pose_image,  # ControlNet 포즈 가이드
                ip_adapter_image=reference_image,  # IP-Adapter 얼굴 참조
                controlnet_conditioning_scale=0.8
            ).images[0]
        else:
            # IP-Adapter 로드 실패 시 기본 ControlNet만 사용
            generated_image = pipe(
                prompt,
                num_inference_steps=20,
                generator=generator,
                image=pose_image
            ).images[0]
    else:
        # reference_image 없으면 기본 ControlNet만 사용
        generated_image = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=pose_image
        ).images[0]
    
    return generated_image