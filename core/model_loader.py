"""모델 로딩 및 관리"""
import asyncio
import torch
from typing import Optional
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from body_analysis_test.body_analysis import BodyAnalysisService

# 전역 변수로 모델 저장
processor = None
model = None

# 새 모델들의 전역 변수 (lazy loading)
segformer_b2_processor = None
segformer_b2_model = None
rtmpose_model = None
realesrgan_model = None
sdxl_pipeline = None

# 체형 분석 서비스 전역 변수
body_analysis_service: Optional[BodyAnalysisService] = None


def _load_segformer_b2_models():
    """SegFormer B2 Human Parsing 모델 로드 (lazy loading)"""
    global segformer_b2_processor, segformer_b2_model
    if segformer_b2_processor is None or segformer_b2_model is None:
        print("SegFormer B2 Human Parsing 모델 로딩 중...")
        segformer_b2_processor = SegformerImageProcessor.from_pretrained("yolo12138/segformer-b2-human-parse-24")
        segformer_b2_model = AutoModelForSemanticSegmentation.from_pretrained("yolo12138/segformer-b2-human-parse-24")
        segformer_b2_model.eval()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        segformer_b2_model = segformer_b2_model.to(device)
        print("SegFormer B2 Human Parsing 모델 로딩 완료")
    return segformer_b2_processor, segformer_b2_model


def _load_rtmpose_model():
    """RTMPose 모델 로드 (lazy loading)"""
    global rtmpose_model
    if rtmpose_model is None:
        try:
            from mmpose.apis import init_model
            config_file = 'configs/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py'
            checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            rtmpose_model = init_model(config_file, checkpoint_file, device=device)
            print("RTMPose-s 모델 로딩 완료!")
        except Exception as e:
            print(f"RTMPose 모델 로딩 실패: {e}")
    return rtmpose_model


def _load_realesrgan_model(scale=4):
    """Real-ESRGAN 모델 로드 (lazy loading)"""
    global realesrgan_model
    if realesrgan_model is None:
        try:
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model_path = f'weights/RealESRGAN_x{scale}plus.pth'
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                   num_conv=32, upscale=scale, act_type='prelu')
            realesrgan_model = RealESRGANer(scale=scale, model_path=model_path, 
                                           model=model, tile=0, tile_pad=10, 
                                           pre_pad=0, half=False, device=device)
            print("Real-ESRGAN 모델 로딩 완료!")
        except Exception as e:
            print(f"Real-ESRGAN 모델 로딩 실패: {e}")
    return realesrgan_model


async def load_models():
    """애플리케이션 시작 시 모델 로드 및 DB 초기화"""
    global processor, model, body_analysis_service
    
    from services.database import init_database
    
    print("SegFormer 모델 로딩 중...")
    # 동기 블로킹 작업을 별도 스레드에서 실행
    loop = asyncio.get_event_loop()
    processor = await loop.run_in_executor(
        None, 
        SegformerImageProcessor.from_pretrained, 
        "mattmdjaga/segformer_b2_clothes"
    )
    model = await loop.run_in_executor(
        None, 
        AutoModelForSemanticSegmentation.from_pretrained, 
        "mattmdjaga/segformer_b2_clothes"
    )
    model.eval()
    print("모델 로딩 완료!")
    
    # DB 초기화
    print("데이터베이스 초기화 중...")
    await loop.run_in_executor(None, init_database)

    # 체형 분석 서비스 초기화
    try:
        print("체형 분석 서비스 초기화 중...")
        body_analysis_service = await loop.run_in_executor(None, BodyAnalysisService)
        if body_analysis_service and body_analysis_service.is_initialized:
            print("✅ 체형 분석 서비스 초기화 완료")
        else:
            print("⚠️  체형 분석 서비스 초기화 실패")
    except Exception as exc:
        print(f"❌ 체형 분석 서비스 로딩 오류: {exc}")
        body_analysis_service = None


def get_processor():
    """전역 processor 반환"""
    return processor


def get_model():
    """전역 model 반환"""
    return model


def get_segformer_b2_processor():
    """전역 segformer_b2_processor 반환 (lazy loading)"""
    if segformer_b2_processor is None:
        _load_segformer_b2_models()
    return segformer_b2_processor


def get_segformer_b2_model():
    """전역 segformer_b2_model 반환 (lazy loading)"""
    if segformer_b2_model is None:
        _load_segformer_b2_models()
    return segformer_b2_model


def get_rtmpose_model():
    """전역 rtmpose_model 반환 (lazy loading)"""
    if rtmpose_model is None:
        _load_rtmpose_model()
    return rtmpose_model


def get_realesrgan_model(scale=4):
    """전역 realesrgan_model 반환 (lazy loading)"""
    if realesrgan_model is None:
        _load_realesrgan_model(scale)
    return realesrgan_model


def get_sdxl_pipeline():
    """전역 sdxl_pipeline 반환 (lazy loading)"""
    global sdxl_pipeline
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
            print(f"SDXL 파이프라인 로딩 실패: {e}")
    return sdxl_pipeline


def get_body_analysis_service():
    """전역 body_analysis_service 반환"""
    return body_analysis_service

