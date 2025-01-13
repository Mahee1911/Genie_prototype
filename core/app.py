from flask import Flask
from flask_cors import CORS


app = Flask(__name__)

# CORS Settings
cors_config = {
    "origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "https://inttent.localhost.one2four.com:4200",
        "https://dev.inttent.app",
        "https://inttent.localhost.one2four.com",
        "prototype-api.dev.inttent.app",
    ],
    "methods": ["*"],
    "allow_headers": ["*", "x-correlation-id"],
    "supports_credentials": True
}

# Apply CORS to all routes
CORS(app, resources={r"/*": cors_config})
