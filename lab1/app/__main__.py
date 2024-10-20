from . import create_app
from .settings import get_settings

import uvicorn


def main() -> None:
    web_app = create_app()
    app_settings = get_settings()

    uvicorn.run(
        app=web_app,
        host=app_settings.host,
        port=app_settings.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
