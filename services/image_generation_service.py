"""Stable Diffusion과 ControlNet을 사용한 이미지 생성 서비스"""
from PIL import Image
from core.model_loader import get_controlnet_pipeline
import torch

async def generate_image_with_pose(prompt: str, pose_image: Image.Image):
    """
    주어진 프롬프트와 포즈 이미지를 사용하여 이미지를 생성합니다.
    """
    # 모델이 필요할 때 getter 함수를 호출하여 로드
    pipe = get_controlnet_pipeline()

    if not pipe:
        raise RuntimeError("이미지 생성 파이프라인을 로드할 수 없습니다.")

    # 입력 이미지 전처리
    pose_image = pose_image.convert("RGB").resize((512, 768)) # SD 1.5는 512x768 같은 해상도도 잘 처리합니다.

    # 재현성을 위한 시드 설정
    generator = torch.manual_seed(0)

    # 이미지 생성 실행
    generated_image = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=pose_image
    ).images[0]
    
    return generated_image