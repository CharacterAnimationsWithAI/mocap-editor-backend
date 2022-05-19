import asyncio
import shutil, os, uuid
from time import sleep, time, strftime, gmtime
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from datastructures.bvh_length_data import BVHLengthData
from datastructures.motion_generation_data import MotionGenerationData
from datastructures.style_transfer_data import StyleTransferData
from cpuinfo import get_cpu_info
import psutil
import GPUtil
from processing.inbetweening.functions import change_bvh_axis
from processing.inbetweening.inbetweening import Inbetweening


from mongodb.api import API
from processing.motion_style_transfer.process_style_transfer import StyleTransfer

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

RESULT_DIRECTORY_NAME = "results"
RESULT_PATH = os.path.join(RESULT_DIRECTORY_NAME)

CPU_INFO = get_cpu_info()['brand_raw'] # takes 1sec


@app.get("/")
async def root():
    return {"status": True}


@app.post("/upload")
async def create_upload_file(file: UploadFile):
    # assigning uuid4
    unique_id = str(uuid.uuid4())
    UPLOAD_FILE = os.path.join(UPLOAD_PATH, unique_id + "-" + file.filename)
    print(UPLOAD_FILE)

    with open(UPLOAD_FILE, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": "upload successful", "filename": unique_id + "-" + file.filename}


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
    return {"statistic": strftime("%H:%M:%S", gmtime(mongo_api.get_average_motion_inference_time()))}


@app.get("/style-transfer-model")
async def get_style_transfer_model_status():
    return {"status": True}


@app.get("/style-transfer-model/inference-time")
async def get_average_style_transfer_time():
    return {"statistic": strftime("%H:%M:%S", gmtime(mongo_api.get_average_style_transfer_time()))}


@app.get("/bvh-length")
async def get_average_bvh_length():
    average_bvh_length = mongo_api.get_average_bvh_length()
    number_of_files = mongo_api.get_number_of_files()
    try:
        average_size = str("{:.2f}".format((average_bvh_length / number_of_files) / (1000000))) + " MB"
    except ZeroDivisionError:
        average_size = "0 MB"

    return {"statistic": average_size}


@app.post("/bvh-length")
async def update_average_bvh_length(data: BVHLengthData):
    mongo_api.update_number_of_files()
    mongo_api.update_average_bvh_length(data.size)
    return {"status": True}



@app.get("/get-logs")
async def get_logs():
    return {"logs": mongo_api.get_logs()}


@app.get("/get-gpu-status")
async def get_gpu_status():
    return {"status": False}


@app.post("/style-transfer-model/inference")
async def apply_style_transfer(style_transfer_data: StyleTransferData):
    # adding time
    start_time = time()

    style_transfer = StyleTransfer(os.path.join(UPLOAD_DIRECTORY_NAME, style_transfer_data.file1), os.path.join(UPLOAD_DIRECTORY_NAME, 
            style_transfer_data.file2), os.path.join(RESULT_PATH, "style_transfer"))

    unique_id = style_transfer.apply_style_transfer()
    
    total_time =  time() - start_time

    # logging request
    # removing uuid4 tag before storing in database
    mongo_api.insert_log({"action": "style_transfer", "motion_generation": False, "source_file": ''.join(style_transfer_data.file1.split('-')[5:]), "target_file": ''.join(style_transfer_data.file2.split('-')[5:]), "date": datetime.now()})
    mongo_api.update_average_style_transfer_time(total_time)

    # return style_transfer_data
    return {"status": True, "url": "http://localhost:8000/result/style-transfer/" + unique_id + "-fixed.bvh"}


@app.get('/system-information')
async def system_information():
    ram_info = "{:.2f}".format(float(psutil.virtual_memory().total) / (1000000000)) + "GB"
    
    gpu_info = None
    vram_info = None

    if GPUtil.getGPUs() == []:
        gpu_info = 'Not Available'
        vram_info = 'Not Availabe'
    else:
        gpu_info = GPUtil.getGPUs()[0].name
        vram_info = str(GPUtil.getGPUs()[0].memoryTotal) + " MB"

    return {"cpu": CPU_INFO, "ram": ram_info, "gpu":gpu_info, "vram":vram_info}


@app.post('/motion-generation-model/inference')
async def motion_generation_model_inference(data: MotionGenerationData):
    fixed_filename = change_bvh_axis(data.filename)
    inbetweening = Inbetweening()
    inbetweening.inbetween(fixed_filename, data.seed_frames, "./results/motion_generation/test.bvh")
    
    return data


@app.get('/result/style-transfer/{filename}')
def get_processing_result(filename):
    return FileResponse(os.path.join(os.path.join(RESULT_PATH, "style_transfer"), filename), media_type="text/plain")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Connection accepted")
    while True:
        # await websocket.send_json({"url": "http://localhost:8000/file/38e0279f-ced7-44db-b6d4-cd3680a13598-fixed.bvh"})
        await asyncio.sleep(5)
        # await websocket.send_json({"url": "http://localhost:8000/file/85f1c481-0637-4ed2-aee1-2e20d07e291c-walk1_subject1.bvh"})
        # await asyncio.sleep(15)
        


# app.mount("/files", StaticFiles(directory=UPLOAD_PATH), name="files")
