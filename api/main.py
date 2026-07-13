"""ASGI entry point for Uvicorn."""

from api.application import create_app


app = create_app()
