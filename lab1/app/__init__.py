from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .services import get_translation_service_startup, get_openai_service_shutdown
from .settings import get_settings
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,  # noqa
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_event_handler("startup", get_translation_service_startup(app))
    app.add_event_handler("shutdown", get_openai_service_shutdown(app))
    
    app.mount("/app", StaticFiles(directory="static", html=True), name="app")
    app.include_router(router, prefix="/api")

    return app
