"""
InstructPix2Pix + GFPGAN 이미지 보정 테스트 스크립트
"""
import torch
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline
from gfpgan import GFPGANer
from pathlib import Path
import argparse

def load_models():
    """모델 로드"""
    print("=" * 50)
    print("모델 로딩 중...")
    print("=" * 50)
    
    # 1. InstructPix2Pix
    print("\n[1/2] InstructPix2Pix 모델 로딩...")
    instruct_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instruct_pipe = instruct_pipe.to(device)
    print(f"✅ InstructPix2Pix 로드 완료 ({device})")
    
    # 2. GFPGAN
    print("\n[2/2] GFPGAN 모델 로딩...")
    model_path = Path("models/GFPGANv1.4.pth")
    
    # 모델이 없으면 다운로드 안내
    if not model_path.exists():
        print(f"⚠️  GFPGAN 모델 파일이 없습니다: {model_path}")
        print("다운로드 URL: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")
        print("또는 자동 다운로드 시도 중...")
        try:
            # 자동 다운로드 시도
            import urllib.request
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            print(f"다운로드 중: {url}")
            urllib.request.urlretrieve(url, str(model_path))
            print("✅ 모델 다운로드 완료")
        except Exception as e:
            print(f"❌ 자동 다운로드 실패: {e}")
            print("수동으로 다운로드 후 models/ 폴더에 저장하세요.")
            gfpgan_restorer = None
    else:
        gfpgan_restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        print(f"✅ GFPGAN 로드 완료")
    
    print("\n" + "=" * 50)
    print("모든 모델 로드 완료!")
    print("=" * 50)
    
    return instruct_pipe, gfpgan_restorer

def translate_korean_instruction(instruction):
    """한국어 요청을 영어 프롬프트로 변환"""
    # 형태 조작
    shape_mappings = {
        "어깨가 너무 넓게": "make shoulders narrower",
        "어깨를 좁게": "make shoulders narrower",
        "어깨를 넓게": "make shoulders wider",
        "허리를 얇게": "make waist thinner",
        "허리를 두껍게": "make waist thicker",
        "엉덩이를 작게": "make hips smaller",
        "엉덩이를 크게": "make hips larger",
    }
    
    # 스타일 변경
    style_mappings = {
        "우아하게": "make it more elegant",
        "캐주얼하게": "change to casual style",
        "모던하게": "make it more modern",
        "빈티지": "change to vintage style",
    }
    
    # 분위기 변경
    mood_mappings = {
        "로맨틱하게": "make the mood more romantic",
        "밝게": "make it bright and cheerful",
        "드라마틱하게": "make it more dramatic",
    }
    
    # 배경 변경
    background_mappings = {
        "배경을 해변으로": "change background to beach",
        "배경을 정원으로": "change background to garden",
        "배경 블러": "make background blur",
        "배경 제거": "remove background",
    }
    
    # 얼굴 보정
    face_mappings = {
        "주름 제거": "remove wrinkles",
        "피부톤 밝게": "make skin tone brighter",
        "피부톤 하얗게": "make skin tone lighter",
        "피부 매끄럽게": "make skin smoother",
    }
    
    # 모든 매핑 병합
    all_mappings = {
        **shape_mappings, 
        **style_mappings, 
        **mood_mappings, 
        **background_mappings,
        **face_mappings
    }
    
    # 매칭된 프롬프트 수집
    prompt_parts = []
    for kor, eng in all_mappings.items():
        if kor in instruction:
            prompt_parts.append(eng)
    
    if prompt_parts:
        return ", ".join(prompt_parts) + " and make it more natural"
    else:
        return f"adjust {instruction} to make it more natural and realistic"

def enhance_image(image_path, instruction, output_path=None, use_gfpgan=True, gfpgan_weight=0.5):
    """
    이미지 보정 파이프라인
    
    Args:
        image_path: 입력 이미지 경로
        instruction: 사용자 요청 (한국어 가능)
        output_path: 출력 이미지 경로 (None이면 자동 생성)
        use_gfpgan: GFPGAN 사용 여부
        gfpgan_weight: GFPGAN 보정 강도 (0.0-1.0)
    """
    print("\n" + "=" * 50)
    print("이미지 보정 시작")
    print("=" * 50)
    print(f"입력 이미지: {image_path}")
    print(f"요청사항: {instruction}")
    
    # 모델 로드
    instruct_pipe, gfpgan_restorer = load_models()
    
    # 이미지 로드
    print("\n이미지 로딩 중...")
    image = Image.open(image_path).convert("RGB")
    print(f"이미지 크기: {image.size}")
    
    # 1단계: InstructPix2Pix (형태/스타일/배경 변경)
    needs_shape_edit = any(keyword in instruction for keyword in 
                          ["어깨", "허리", "엉덩이", "스타일", "배경", "분위기", "로맨틱", "밝게", "드라마틱"])
    
    if needs_shape_edit:
        print("\n[1단계] InstructPix2Pix로 편집 중...")
        prompt = translate_korean_instruction(instruction)
        print(f"프롬프트: {prompt}")
        
        result = instruct_pipe(
            prompt,
            image=image,
            num_inference_steps=20,
            image_guidance_scale=1.5
        ).images[0]
        
        print("✅ InstructPix2Pix 편집 완료")
    else:
        result = image
        print("⚠️  형태/스타일 변경이 필요하지 않아 InstructPix2Pix 단계를 건너뜁니다.")
    
    # 2단계: GFPGAN (얼굴 보정)
    needs_face_retouch = any(keyword in instruction for keyword in 
                            ["주름", "피부", "톤", "얼굴", "보정", "하얗게", "밝게"])
    
    if use_gfpgan and needs_face_retouch and gfpgan_restorer is not None:
        print("\n[2단계] GFPGAN으로 얼굴 보정 중...")
        
        # PIL → OpenCV 변환
        img_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 보정 강도 조절
        if "밝게" in instruction or "하얗게" in instruction:
            weight = min(gfpgan_weight + 0.2, 0.8)  # 더 강하게
        elif "자연스럽게" in instruction:
            weight = max(gfpgan_weight - 0.2, 0.3)  # 더 자연스럽게
        else:
            weight = gfpgan_weight
        
        print(f"보정 강도: {weight:.2f}")
        
        _, _, img_cv = gfpgan_restorer.enhance(
            img_cv,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight
        )
        
        # OpenCV → PIL 변환
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        print("✅ GFPGAN 보정 완료")
    elif needs_face_retouch and gfpgan_restorer is None:
        print("⚠️  GFPGAN 모델이 없어 얼굴 보정을 건너뜁니다.")
    
    # 출력 경로 설정
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    
    # 결과 저장
    result.save(output_path)
    print(f"\n✅ 이미지 보정 완료!")
    print(f"출력 경로: {output_path}")
    print("=" * 50)
    
    return result, str(output_path)

def main():
    parser = argparse.ArgumentParser(description="이미지 보정 테스트 (InstructPix2Pix + GFPGAN)")
    parser.add_argument("image", type=str, help="입력 이미지 경로")
    parser.add_argument("instruction", type=str, help="보정 요청사항 (한국어 가능)")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 이미지 경로")
    parser.add_argument("--no-gfpgan", action="store_true", help="GFPGAN 사용 안 함")
    parser.add_argument("--gfpgan-weight", type=float, default=0.5, help="GFPGAN 보정 강도 (0.0-1.0)")
    
    args = parser.parse_args()
    
    try:
        result, output_path = enhance_image(
            args.image,
            args.instruction,
            args.output,
            use_gfpgan=not args.no_gfpgan,
            gfpgan_weight=args.gfpgan_weight
        )
        print(f"\n✅ 성공! 결과 이미지: {output_path}")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 예시 사용법
    if len(__import__("sys").argv) == 1:
        print("=" * 50)
        print("이미지 보정 테스트 스크립트")
        print("=" * 50)
        print("\n사용법:")
        print("  python test_image_enhancement.py <이미지경로> <요청사항>")
        print("\n예시:")
        print("  python test_image_enhancement.py test.jpg \"어깨 좁게, 주름 제거, 피부톤 밝게\"")
        print("  python test_image_enhancement.py test.jpg \"우아한 스타일로, 배경 블러\"")
        print("  python test_image_enhancement.py test.jpg \"로맨틱한 분위기로\" -o result.png")
        print("\n옵션:")
        print("  -o, --output: 출력 경로 지정")
        print("  --no-gfpgan: GFPGAN 사용 안 함")
        print("  --gfpgan-weight: GFPGAN 보정 강도 (0.0-1.0, 기본값: 0.5)")
        print("\n" + "=" * 50)
    else:
        main()






