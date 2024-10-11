from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import detection

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(detection.router)


@app.get("/")
async def root():
    return {"message": "Yolo Server is running"}
