from core.app import app
from . import upload

app.include_router(upload.router, prefix="/api", tags=["upload"])