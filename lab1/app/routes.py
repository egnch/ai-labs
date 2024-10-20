from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .services import (
    TranslationService,
    depends_translation_service,
    Language,
    TranslationServiceType,
)


router = APIRouter()


class TranslateRequest(BaseModel):
    text: str
    from_language: Language
    to_language: Language
    service: TranslationServiceType


class TranslateResponse(BaseModel):
    translation: str


@router.get("/test")
async def test():
    return "test"

@router.post("/translate")
async def translate(
    request: TranslateRequest,
    service: TranslationService = Depends(depends_translation_service),
):
    translation = await service.translate(
        request.text, request.from_language, request.to_language, request.service
    )
    return TranslateResponse(translation=translation)
