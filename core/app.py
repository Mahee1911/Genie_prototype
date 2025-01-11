from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=["https://dev.inttent.app", "https://inttent.localhost.one2four.com", "prototype-api.dev.inttent.app", "http://54.68.56.173/"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

@app.get("/")
async def ping():
    return {"status": "ok", "message": "pong"}

