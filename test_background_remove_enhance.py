"""
배경 분위기 변경 + 보정 모델 테스트 스크립트
1단계: InstructPix2Pix (형태/스타일/배경 분위기 변경)
2단계: GFPGAN (얼굴 보정)
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
    """모든 모델 로드"""
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60)
    
    models = {}
    
    # 1. InstructPix2Pix (형태/스타일/배경 분위기 변경)
    print("\n[1/2] InstructPix2Pix 모델 로딩 (형태/스타일/배경 분위기 변경)...")
    try:
        models['instruct_pipe'] = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models['instruct_pipe'] = models['instruct_pipe'].to(device)
        print(f"✅ InstructPix2Pix 로드 완료 ({device})")
    except Exception as e:
        print(f"❌ InstructPix2Pix 로드 실패: {e}")
        models['instruct_pipe'] = None
    
    # 2. GFPGAN (얼굴 보정)
    print("\n[2/2] GFPGAN 모델 로딩 (얼굴 보정)...")
    model_path = Path("models/GFPGANv1.4.pth")
    
    if not model_path.exists():
        print(f"⚠️  GFPGAN 모델 파일이 없습니다: {model_path}")
        print("다운로드 URL: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")
        try:
            import urllib.request
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            print(f"다운로드 중: {url}")
            urllib.request.urlretrieve(url, str(model_path))
            print("✅ 모델 다운로드 완료")
        except Exception as e:
            print(f"❌ 자동 다운로드 실패: {e}")
            models['gfpgan'] = None
    else:
        try:
            models['gfpgan'] = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print(f"✅ GFPGAN 로드 완료")
        except Exception as e:
            print(f"❌ GFPGAN 로드 실패: {e}")
            models['gfpgan'] = None
    
    print("\n" + "=" * 60)
    print("모델 로딩 완료!")
    print("=" * 60)
    
    return models

def enhance_image(image, instruction, models, use_gfpgan=True, gfpgan_weight=0.5):
    """
    이미지 보정 (형태/스타일/배경 분위기 변경 + 얼굴 보정)
    
    Args:
        image: PIL Image (RGB)
        instruction: 보정 요청사항
        models: 로드된 모델 딕셔너리
        use_gfpgan: GFPGAN 사용 여부
        gfpgan_weight: GFPGAN 보정 강도
    """
    
    # 1단계: InstructPix2Pix (형태/스타일/배경 분위기 변경)
    needs_edit = any(keyword in instruction for keyword in 
                    ["어깨", "허리", "엉덩이", "스타일", "배경", "분위기", "로맨틱", "밝게", "드라마틱", "조명", "색감"])
    
    if needs_edit and models['instruct_pipe'] is not None:
        print("\n[1단계] InstructPix2Pix로 보정 중...")
        prompt = translate_instruction(instruction)
        print(f"프롬프트: {prompt}")
        
        result = models['instruct_pipe'](
            prompt,
            image=image,
            num_inference_steps=20,
            image_guidance_scale=1.5
        ).images[0]
        print("✅ InstructPix2Pix 보정 완료")
    else:
        result = image
        if not needs_edit:
            print("⚠️  보정 요청사항이 없어 InstructPix2Pix 단계를 건너뜁니다.")
        else:
            print("⚠️  InstructPix2Pix 모델이 없어 보정을 건너뜁니다.")
    
    # 2단계: GFPGAN (얼굴 보정)
    needs_face_retouch = any(keyword in instruction for keyword in 
                            ["주름", "피부", "톤", "얼굴", "보정", "하얗게", "밝게"])
    
    if use_gfpgan and needs_face_retouch and models['gfpgan'] is not None:
        print("\n[3단계] GFPGAN으로 얼굴 보정 중...")
        
        # PIL → OpenCV
        img_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 보정 강도 조절
        if "밝게" in instruction or "하얗게" in instruction:
            weight = min(gfpgan_weight + 0.2, 0.8)
        elif "자연스럽게" in instruction:
            weight = max(gfpgan_weight - 0.2, 0.3)
        else:
            weight = gfpgan_weight
        
        print(f"보정 강도: {weight:.2f}")
        
        _, _, img_cv = models['gfpgan'].enhance(
            img_cv,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight
        )
        
        # OpenCV → PIL
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        print("✅ GFPGAN 보정 완료")
    elif needs_face_retouch and models['gfpgan'] is None:
        print("⚠️  GFPGAN 모델이 없어 얼굴 보정을 건너뜁니다.")
    
    return result

def translate_instruction(instruction):
    """한국어 요청을 영어 프롬프트로 변환"""
    # 형태 조작
    shape_mappings = {
        "어깨가 너무 넓게": "make shoulders narrower",
        "어깨를 좁게": "make shoulders narrower",
        "허리를 얇게": "make waist thinner",
        "엉덩이를 작게": "make hips smaller",
    }
    
    # 스타일 변경
    style_mappings = {
        "우아하게": "make it more elegant",
        "모던하게": "make it more modern",
        "캐주얼하게": "change to casual style",
        "빈티지": "change to vintage style",
    }
    
    # 배경 분위기 변경
    background_mappings = {
        "배경 블러": "make background blur",
        "배경을 해변으로": "change background to beach",
        "배경을 정원으로": "change background to garden",
        "배경을 스튜디오로": "change background to studio",
        "배경을 교회로": "change background to church",
        "배경을 공원으로": "change background to park",
    }
    
    # 분위기 변경
    mood_mappings = {
        "로맨틱하게": "make the mood more romantic",
        "밝게": "make it bright and cheerful",
        "드라마틱하게": "make it more dramatic",
        "따뜻하게": "change to warm and cozy feeling",
        "고급스럽게": "make it more luxurious",
    }
    
    # 조명/색감
    lighting_mappings = {
        "조명을 부드럽게": "make lighting more soft",
        "따뜻한 조명으로": "change to warm lighting",
        "밝은 조명으로": "change to bright lighting",
        "색감을 선명하게": "make colors more vibrant",
    }
    
    # 모든 매핑 병합
    all_mappings = {
        **shape_mappings,
        **style_mappings,
        **background_mappings,
        **mood_mappings,
        **lighting_mappings
    }
    
    prompt_parts = []
    for kor, eng in all_mappings.items():
        if kor in instruction:
            prompt_parts.append(eng)
    
    if prompt_parts:
        return ", ".join(prompt_parts) + " and make it more natural"
    else:
        return f"adjust {instruction} to make it more natural and realistic"

def process_image(image_path, instruction, output_path=None, use_gfpgan=True, gfpgan_weight=0.5):
    """
    전체 파이프라인: 배경 분위기 변경 + 보정
    
    Args:
        image_path: 입력 이미지 경로
        instruction: 보정 요청사항 (형태/스타일/배경 분위기/얼굴 보정)
        output_path: 출력 이미지 경로
        use_gfpgan: GFPGAN 사용 여부
        gfpgan_weight: GFPGAN 보정 강도
    """
    print("\n" + "=" * 60)
    print("배경 분위기 변경 + 보정 파이프라인 시작")
    print("=" * 60)
    print(f"입력 이미지: {image_path}")
    print(f"요청사항: {instruction}")
    
    # 모델 로드
    models = load_models()
    
    # 이미지 로드
    print("\n이미지 로딩 중...")
    image = Image.open(image_path).convert("RGB")
    print(f"이미지 크기: {image.size}")
    
    # 보정 처리
    if instruction:
        result = enhance_image(image, instruction, models, use_gfpgan, gfpgan_weight)
    else:
        print("\n⚠️  보정 요청사항이 없습니다.")
        result = image
    
    # 출력 경로 설정
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    
    # 결과 저장
    result.save(output_path)
    print(f"\n✅ 처리 완료!")
    print(f"출력 경로: {output_path}")
    print("=" * 60)
    
    return result, str(output_path)

def main():
    parser = argparse.ArgumentParser(description="배경 분위기 변경 + 보정 테스트")
    parser.add_argument("image", type=str, help="입력 이미지 경로")
    parser.add_argument("-i", "--instruction", type=str, default="", help="보정 요청사항 (한국어 가능)")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 이미지 경로")
    parser.add_argument("--no-gfpgan", action="store_true", help="GFPGAN 사용 안 함")
    parser.add_argument("--gfpgan-weight", type=float, default=0.5, help="GFPGAN 보정 강도 (0.0-1.0)")
    
    args = parser.parse_args()
    
    try:
        result, output_path = process_image(
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
        print("=" * 60)
        print("배경 분위기 변경 + 보정 테스트 스크립트")
        print("=" * 60)
        print("\n사용법:")
        print("  python test_background_remove_enhance.py <이미지경로> [옵션]")
        print("\n예시:")
        print("  # 형태 + 배경 분위기 + 얼굴 보정")
        print("  python test_background_remove_enhance.py test.jpg -i \"어깨 좁게, 배경 블러, 로맨틱한 분위기, 주름 제거\"")
        print("\n  # 배경 분위기만 변경")
        print("  python test_background_remove_enhance.py test.jpg -i \"배경을 해변으로, 밝은 조명으로\" --no-gfpgan")
        print("\n  # 스타일 + 배경 분위기")
        print("  python test_background_remove_enhance.py test.jpg -i \"우아한 스타일로, 배경을 정원으로, 따뜻한 조명\"")
        print("\n  # 출력 경로 지정")
        print("  python test_background_remove_enhance.py test.jpg -i \"로맨틱한 분위기로\" -o result.png")
        print("\n지원하는 요청사항:")
        print("  - 형태: 어깨 좁게, 허리 얇게, 엉덩이 작게")
        print("  - 스타일: 우아하게, 모던하게, 캐주얼하게")
        print("  - 배경 분위기: 배경 블러, 배경을 해변/정원/교회/스튜디오로")
        print("  - 분위기: 로맨틱하게, 밝게, 드라마틱하게, 따뜻하게")
        print("  - 조명/색감: 조명 부드럽게, 따뜻한 조명, 색감 선명하게")
        print("  - 얼굴 보정: 주름 제거, 피부톤 밝게")
        print("\n옵션:")
        print("  -i, --instruction: 보정 요청사항 (한국어 가능)")
        print("  -o, --output: 출력 경로 지정")
        print("  --no-gfpgan: GFPGAN 사용 안 함 (얼굴 보정 제외)")
        print("  --gfpgan-weight: GFPGAN 보정 강도 (0.0-1.0, 기본값: 0.5)")
        print("\n" + "=" * 60)
    else:
        main()

