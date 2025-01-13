from flask import Flask
from flask_cors import CORS


app = Flask(__name__)

# CORS Settings
cors_config = {
    "origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4200",
        "https://dev.inttent.app",
        "https://inttent.localhost.one2four.com",
        "prototype-api.dev.inttent.app",
    ],
    "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
    "expose_headers": ["Content-Range", "X-Content-Range"],
    "supports_credentials": True,
    "max_age": 600,  # Cache preflight requests for 10 minutes
    "send_wildcard": False,
}

# Apply CORS to all routes
CORS(app, resources={r"/*": cors_config})
