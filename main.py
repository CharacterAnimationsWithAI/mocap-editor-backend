import shutil, os
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:1212"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY_NAME = "uploads"
UPLOAD_PATH = os.path.join(UPLOAD_DIRECTORY_NAME)

@app.get("/")
async def root():
    return {"message": "Hello, World"}


@app.post("/upload")
async def create_upload_file(file: UploadFile):
    UPLOAD_FILE = os.path.join(UPLOAD_PATH, file.filename)
    print(UPLOAD_FILE)
    with open(UPLOAD_FILE, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": "upload successful", "filename": file.filename}


@app.get("/file/{filename}")
async def get_file(filename):
    return FileResponse(os.path.join(UPLOAD_PATH, filename), media_type="text/plain")
 

# app.mount("/files", StaticFiles(directory=UPLOAD_PATH), name="files")
