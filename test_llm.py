import base64
import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
from jinja2 import TemplateNotFound
from openai import OpenAI
from pydantic import BaseModel
from starlette.templating import Jinja2Templates


logger = logging.getLogger("test_llm")
logging.basicConfig(level=logging.INFO)


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class APISettings(BaseModel):
    openai_model: str = "gpt-4o"
    gemini_model: str = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.0-flash")


settings = APISettings()

app = FastAPI(
    title="LLM Composition Test",
    description="GPT-4o 기반 프롬프트 생성 및 Gemini 2.x Flash 이미지 합성 테스트용 독립 서버",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"환경변수 '{key}'가 설정되어 있지 않습니다.",
        )
    return value


def _bytes_to_data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _build_openai_prompt_inputs(person_b64: str, dress_b64: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [
            {
                "type": "text",
                "text": (
                    "You are a professional visual prompt engineer specialized in AI outfit try-on. "
                    "Your job is to analyze two reference images: (1) a person photo, and (2) a clothing photo. "
                    "Then, write a detailed English prompt for a generative image model (e.g., Gemini 2.5 Flash) "
                    "that will replace only the person's outfit with the clothing from the second image. "
                    "Rules: Keep the same person's face, body shape, pose, hairstyle, and background exactly. "
                    "Do NOT change facial expression, body proportions, or lighting. "
                    "Describe the new outfit (color, texture, fabric, style) based on the clothing image. "
                    "Make it photorealistic and naturally blended, as if the person was originally photographed wearing it. "
                    "Return ONLY the final prompt text, no explanations."
                ),
            }
        ]},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Analyze the outfit replacement request."},
            {"type": "input_image", "image_base64": person_b64},
            {"type": "input_text", "text": "Clothing reference image:"},
            {"type": "input_image", "image_base64": dress_b64},
        ]},
    ]


def _extract_openai_prompt(response: Any) -> str:
    try:
        text = response.output_text  # type: ignore[attr-defined]
    except AttributeError as exc:  # pragma: no cover - defensive
        logger.error("Failed to parse OpenAI response: %s", exc)
        raise HTTPException(status_code=500, detail="GPT-4o 응답 파싱에 실패했습니다.")

    prompt = (text or "").strip()
    if not prompt:
        raise HTTPException(status_code=500, detail="GPT-4o가 유효한 프롬프트를 반환하지 않았습니다.")
    return prompt


def _extract_gemini_image(response: Any) -> Dict[str, str]:
    try:
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.mime_type.startswith("image/"):
                return {
                    "mime_type": inline.mime_type,
                    "data": inline.data,
                }
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to parse Gemini response: %s", exc)
        raise HTTPException(status_code=500, detail="Gemini 응답 파싱에 실패했습니다.")
    raise HTTPException(status_code=500, detail="Gemini 응답에 이미지가 포함되어 있지 않습니다.")


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request) -> HTMLResponse:
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except TemplateNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/model-comparison", response_class=HTMLResponse)
async def serve_model_comparison(request: Request) -> HTMLResponse:
    try:
        return templates.TemplateResponse("model-comparison.html", {"request": request})
    except TemplateNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/models")
async def list_models() -> JSONResponse:
    models = [
        {
            "id": "gpt4o-gemini",
            "name": "GPT-4o → Gemini 2.5 Flash V1",
            "description": "GPT-4o가 생성한 프롬프트로 Gemini 2.5 Flash가 이미지를 합성합니다.",
            "endpoint": "/api/gpt4o-gemini/compose",
            "method": "POST",
            "input_type": "dual_image",
            "inputs": [
                {"name": "person_image", "label": "사람 이미지", "required": True},
                {"name": "dress_image", "label": "드레스 이미지", "required": True},
            ],
            "category": "composition",
            "requires_prompt_generation": True,
        }
    ]
    return JSONResponse({"success": True, "models": models})


@app.post("/api/generate-prompt")
async def generate_prompt(
    person_image: UploadFile = File(...),
    dress_image: UploadFile = File(...),
) -> JSONResponse:
    openai_api_key = _require_env("OPENAI_API_KEY")

    person_bytes = await person_image.read()
    dress_bytes = await dress_image.read()

    if not person_bytes or not dress_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일을 모두 업로드해주세요.")

    person_b64 = base64.b64encode(person_bytes).decode("utf-8")
    dress_b64 = base64.b64encode(dress_bytes).decode("utf-8")

    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=_build_openai_prompt_inputs(person_b64, dress_b64),
            max_output_tokens=500,
        )
    except Exception as exc:
        logger.exception("OpenAI API 호출 실패: %s", exc)
        raise HTTPException(status_code=502, detail=f"GPT-4o 호출 실패: {exc}")

    prompt = _extract_openai_prompt(response)
    logger.info("Prompt generated successfully (%d characters)", len(prompt))

    return JSONResponse({"success": True, "prompt": prompt})


@app.post("/api/gpt4o-gemini/compose")
async def compose_image(
    person_image: UploadFile = File(...),
    dress_image: UploadFile = File(...),
    prompt: str = "",
) -> JSONResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt 필드는 비어 있을 수 없습니다.")

    google_api_key = _require_env("GOOGLE_API_KEY")
    genai.configure(api_key=google_api_key)

    person_bytes = await person_image.read()
    dress_bytes = await dress_image.read()

    if not person_bytes or not dress_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일을 모두 업로드해주세요.")

    model = genai.GenerativeModel(model_name=settings.gemini_model)

    try:
        response = model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": person_image.content_type or "image/jpeg",
                                "data": base64.b64encode(person_bytes).decode("utf-8"),
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": dress_image.content_type or "image/jpeg",
                                "data": base64.b64encode(dress_bytes).decode("utf-8"),
                            }
                        },
                    ],
                }
            ],
            generation_config={
                "temperature": 0.3,
            },
        )
    except google_exceptions.GoogleAPIError as exc:
        logger.exception("Gemini API 호출 실패: %s", exc)
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Gemini API 알 수 없는 오류: %s", exc)
        raise HTTPException(status_code=500, detail=f"Gemini 호출 중 알 수 없는 오류가 발생했습니다: {exc}")

    image_payload = _extract_gemini_image(response)
    result_data_url = f"data:{image_payload['mime_type']};base64,{image_payload['data']}"

    person_data_url = _bytes_to_data_url(person_bytes, person_image.content_type or "image/jpeg")
    dress_data_url = _bytes_to_data_url(dress_bytes, dress_image.content_type or "image/jpeg")

    return JSONResponse(
        {
            "success": True,
            "result_image": result_data_url,
            "person_image": person_data_url,
            "dress_image": dress_data_url,
        }
    )


@app.get("/healthz")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "test_llm:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8004)),
        reload=True,
    )

