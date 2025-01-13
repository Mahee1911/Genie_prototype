from core.app import app
from . import upload

# Register the blueprint with the prefix
app.register_blueprint(upload.router)
