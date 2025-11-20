import logging
import torch
import mediapipe as mp
import os
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from body_analysis_test.body_analysis import BodyAnalysisService
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

segformer_b2_processor = None
segformer_b2_model = None
rtmpose_model = None
realesrgan_model = None
sdxl_pipeline = None
controlnet_pipeline = None
body_analysis_service = None
mediapipe_pose_model = None

def load_segformer_b2_models():
    global segformer_b2_processor, segformer_b2_model
    if segformer_b2_processor is None or segformer_b2_model is None:
        try:
            logger.info("SegFormer B2 Human Parsing 모델 로딩 중...")
            model_name = "yolo12138/segformer-b2-human-parse-24"
            segformer_b2_processor = SegformerImageProcessor.from_pretrained(model_name)
            segformer_b2_model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            segformer_b2_model.to(device).eval()
            logger.info("SegFormer B2 Human Parsing 모델 로딩 완료")
        except Exception as e:
            logger.error(f"SegFormer B2 Human Parsing 모델 로딩 실패: {e}")
            segformer_b2_processor, segformer_b2_model = None, None
    return segformer_b2_processor, segformer_b2_model

def get_segformer_b2_processor():
    if segformer_b2_processor is None: load_segformer_b2_models()
    return segformer_b2_processor

def get_segformer_b2_model():
    if segformer_b2_model is None: load_segformer_b2_models()
    return segformer_b2_model

def load_rtmpose_model():
    global rtmpose_model
    if rtmpose_model is None:
        try:
            logger.info("RTMPose 모델 로딩 중...")
            from mmpose.apis import init_model
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
            config_file = os.path.join(project_root, 'config', 'rtmpose', 'rtmpose-s_8xb256-420e_coco-256x192.py')
            if not os.path.exists(config_file):
                logger.error(f"RTMPose 설정 파일을 찾을 수 없습니다: {config_file}")
                raise FileNotFoundError(f"RTMPose 설정 파일을 찾을 수 없습니다: {config_file}")
            checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            rtmpose_model = init_model(config_file, checkpoint_file, device=device)
            logger.info("RTMPose 모델 로딩 완료!")
        except Exception as e:
            logger.error(f"RTMPose 모델 로딩 실패: {e}")
            rtmpose_model = None
    return rtmpose_model

def get_rtmpose_model():
    if rtmpose_model is None: load_rtmpose_model()
    return rtmpose_model

def load_realesrgan_model(scale=4):
    global realesrgan_model
    if realesrgan_model is None:
        try:
            logger.info(f"Real-ESRGAN 모델 로딩 중 (scale={scale})...")
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
            model_path = os.path.join(project_root, 'weights', f'RealESRGAN_x{scale}plus.pth')
            if not os.path.exists(model_path):
                logger.error(f"Real-ESRGAN 가중치 파일을 찾을 수 없습니다: {model_path}")
                raise FileNotFoundError(f"Real-ESRGAN 가중치 파일을 찾을 수 없습니다: {model_path}")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            realesrgan_model = RealESRGANer(scale=scale, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True if device.startswith('cuda') else False, device=device)
            logger.info(f"Real-ESRGAN 모델 로딩 완료 (scale={scale})")
        except Exception as e:
            logger.error(f"Real-ESRGAN 모델 로딩 실패: {e}")
            realesrgan_model = None
    return realesrgan_model

def load_sdxl_pipeline():
    global sdxl_pipeline
    if sdxl_pipeline is None:
        try:
            logger.info("SDXL 파이프라인 로딩 중...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32)
            if device.startswith('cuda'):
                sdxl_pipeline.enable_model_cpu_offload()
            else:
                sdxl_pipeline.to(device)
            logger.info("SDXL 파이프라인 로딩 완료!")
        except Exception as e:
            logger.error(f"SDXL 파이프라인 로딩 실패: {repr(e)}")
            sdxl_pipeline = None
    return sdxl_pipeline

def get_sdxl_pipeline():
    if sdxl_pipeline is None: load_sdxl_pipeline()
    return sdxl_pipeline

def load_controlnet_pipeline():
    global controlnet_pipeline
    if controlnet_pipeline is None:
        try:
            logger.info("ControlNet 파이프라인 로딩 중...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
            pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            if device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
            controlnet_pipeline = pipe
            logger.info("ControlNet 파이프라인 로딩 완료!")
        except Exception as e:
            logger.error(f"ControlNet 파이프라인 로딩 실패: {repr(e)}", exc_info=True)
            controlnet_pipeline = None
    return controlnet_pipeline

def get_controlnet_pipeline():
    if controlnet_pipeline is None: load_controlnet_pipeline()
    return controlnet_pipeline

def get_mediapipe_pose():
    global mediapipe_pose_model
    if mediapipe_pose_model is None:
        logger.info("MediaPipe Pose 모델 로딩 중...")
        mediapipe_pose_model = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        logger.info("MediaPipe Pose 모델 로딩 완료")
    return mediapipe_pose_model, mp.solutions.drawing_utils

def load_body_analysis_service():
    global body_analysis_service
    if body_analysis_service is None:
        try:
            logger.info("BodyAnalysis 서비스 로딩 중...")
            current_file_path = os.path.abspath(__file__)
            core_dir = os.path.dirname(current_file_path)
            model_path_str = os.path.join(core_dir, '..', 'model', 'pose_model.pth')
            absolute_model_path = os.path.abspath(model_path_str)
            if not os.path.exists(absolute_model_path):
                logger.error(f"BodyAnalysis 모델 파일을 찾을 수 없습니다: {absolute_model_path}")
                body_analysis_service = None
                return None
            body_analysis_service = BodyAnalysisService(model_path=absolute_model_path)
            logger.info("BodyAnalysis 서비스 로딩 완료!")
        except Exception as e:
            logger.error(f"BodyAnalysis 서비스 로딩 실패: {e}", exc_info=True)
            body_analysis_service = None
    return body_analysis_service

def load_all_models():
    logger.info("모든 모델 사전 로딩 시작...")
    load_segformer_b2_models()
    load_rtmpose_model()
    load_realesrgan_model()
    load_controlnet_pipeline()
    get_mediapipe_pose()
    load_body_analysis_service()
    logger.info("모든 모델 사전 로딩 완료 (SDXL은 on-demand).")
