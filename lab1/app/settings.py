from functools import cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_key: str
    rapidapi_key: str
    yandex_key: str

    host: str = "0.0.0.0"
    port: int = 8000


@cache
def get_settings():
    return Settings()
