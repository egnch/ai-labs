from typing import Awaitable, Callable

from aiohttp import ClientSession
from fastapi import FastAPI, Request
from openai import AsyncClient
from .settings import get_settings
from enum import StrEnum


class Language(StrEnum):
    RU = "ru"
    EN = "en"


class TranslationServiceType(StrEnum):
    DEEP = "deep-translate"
    GOOGLE = "google-translate"
    YANDEX = "yandex-translate"
    GPT_4O = "openai-gpt-4o"
    GPT_4O_MINI = "openai-gpt-4o-mini"
    GPT_3_5_TURBO = "openai-gpt-3.5-turbo"


class TranslationService:
    yandex_session: ClientSession
    rapidapi_session: ClientSession
    openai_client: AsyncClient
    services: dict[TranslationServiceType, Callable[[str, Language, Language], Awaitable[str]]]

    def __init__(self, openai_key: str, rapidapi_key: str, yandex_key: str):
        self.openai_client = AsyncClient(api_key=openai_key)
        self.rapidapi_session = ClientSession(headers={"x-rapidapi-key": rapidapi_key})
        self.yandex_session = ClientSession(
            headers={"Authorization": f"Api-Key {yandex_key}"}
        )

        self.services = {
            TranslationServiceType.DEEP: self.translate_deep,
            TranslationServiceType.GOOGLE: self.translate_google,
            TranslationServiceType.YANDEX: self.translate_yandex,
            TranslationServiceType.GPT_4O: self.translate_gpt_4o,
            TranslationServiceType.GPT_4O_MINI: self.translate_gpt_4o_mini,
            TranslationServiceType.GPT_3_5_TURBO: self.translate_gpt_3_5_turbo,
        }

    async def close(self):
        await self.yandex_session.close()
        await self.rapidapi_session.close()
        await self.openai_client.close()

    async def translate_deep(
        self, text: str, from_language: Language, to_language: Language
    ) -> str:
        url = "https://deep-translate1.p.rapidapi.com/language/translate/v2"

        headers = {"x-rapidapi-host": "deep-translate1.p.rapidapi.com"}
        payload = {"q": text, "source": from_language, "target": to_language}

        async with self.rapidapi_session.post(
            url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["data"]["translations"]["translatedText"]

    async def translate_google(
        self, text: str, from_language: Language, to_language: Language
    ) -> str:
        url = "https://google-translate113.p.rapidapi.com/api/v1/translator/text"
        headers = {"x-rapidapi-host": "google-translate113.p.rapidapi.com"}
        payload = {"from": from_language, "to": to_language, "text": text}

        async with self.rapidapi_session.post(
            url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["trans"]

    async def translate_yandex(
        self, text: str, from_language: Language, to_language: Language
    ) -> str:
        url = "https://translate.api.cloud.yandex.net/translate/v2/translate"

        payload = {
            "text": text,
            "sourceLanguageCode": from_language,
            "targetLanguageCode": to_language,
            "texts": [text],
        }

        async with self.yandex_session.post(url, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["translations"][0]["text"]

    async def translate_gpt(
        self, text: str, model: str, from_language: Language, to_language: Language
    ) -> str:
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        "Translate text from one language to another. "
                        "Answer only with translation and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Translate the following text from {from_language} to {to_language}:\n"
                        f"{text}"
                    ),
                },
            ],
            temperature=0.5,
        )

        return response.choices[0].message.content

    async def translate_gpt_4o(
        self, text: str, from_language: Language, to_language: Language
    ) -> str:
        return await self.translate_gpt(text, "gpt-4o", from_language, to_language)

    async def translate_gpt_4o_mini(
        self, text: str, from_language: Language, to_language: Language
    ) -> str:
        return await self.translate_gpt(text, "gpt-4o-mini", from_language, to_language)

    async def translate_gpt_3_5_turbo(
        self, text: str, from_language: Language, to_language: Language
    ) -> str:
        return await self.translate_gpt(
            text, "gpt-3.5-turbo", from_language, to_language
        )

    async def translate(
        self,
        text: str,
        from_language: Language,
        to_language: Language,
        service: TranslationServiceType,
    ) -> str:
        func = self.services.get(service)
        if func is None:
            raise ValueError(f"Unknown translation service: {service}")

        return await func(text, from_language, to_language)


def get_translation_service_startup(app: FastAPI) -> Callable[[], Awaitable[None]]:
    async def translation_service_startup() -> None:
        settings = get_settings()
        app.state.translation_service = TranslationService(
            settings.openai_key, settings.rapidapi_key, settings.yandex_key
        )

    return translation_service_startup


def get_openai_service_shutdown(app: FastAPI) -> Callable[[], Awaitable[None]]:
    async def openai_service_shutdown():
        await app.state.translation_service.close()

    return openai_service_shutdown


async def depends_translation_service(request: Request) -> TranslationService:
    return request.app.state.translation_service
