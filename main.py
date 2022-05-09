import shutil, os
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datastructures.inference_data import InferenceData

from mongodb.api import API

# MongoDB driver for database queries
mongo_api = API("localhost:27017")
# mongo_api.insert_log({"action": "motion_generation", "motion_generation": True, "source_file": "walk1.bvh", "target_file": "st2.bvh", "date": datetime.now()})
# results = mongo_api.get_logs()
# print(results)
# mongo_api.update_average_bvh_length(100.5)
# print(mongo_api.get_average_motion_inference_time())

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

STATIC_DIRECTORY_NAME = "static"
STATIC_PATH = os.path.join(STATIC_DIRECTORY_NAME)

@app.get("/")
async def root():
    return {"status": True}


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


@app.get("/logo")
async def get_logo():
    return FileResponse(os.path.join(STATIC_PATH, "motion-capture.png"), media_type="image/png")
 

@app.get("/blender-server")
async def get_blender_server_status():
    return {"status": False}


@app.get("/motion-generation-model")
async def get_motion_generation_model_status():
    return {"status": False}


@app.get("/motion-generation-model/inference-time")
async def get_average_motion_inference_time():
    return {"statistic": str(timedelta(seconds=mongo_api.get_average_motion_inference_time()))}


@app.get("/style-transfer-model")
async def get_style_transfer_model_status():
    return {"status": False}


@app.get("/style-transfer-model/inference-time")
async def get_average_style_transfer_time():
    return {"statistic": str(timedelta(seconds=mongo_api.get_average_style_transfer_time()))}


@app.get("/bvh-length")
async def get_average_bvh_length():
    return {"statistic": str(timedelta(seconds=mongo_api.get_average_bvh_length()))}


@app.get("/get-logs")
async def get_logs():
    return {"logs": mongo_api.get_logs()}


@app.get("/get-gpu-status")
async def get_gpu_status():
    return {"status": False}


@app.post("/style-transfer-model/inference")
async def apply_style_transfer(inference_data: InferenceData):
    ### TODO: add style transfer model code

    # logging request
    mongo_api.insert_log({"action": "style_transfer", "motion_generation": False, "source_file": inference_data.file1, "target_file": inference_data.file2, "date": datetime.now()})

    return inference_data

# app.mount("/files", StaticFiles(directory=UPLOAD_PATH), name="files")
