import asyncio
import contextlib
import os
import threading
import time
from typing import List, Optional, Dict, Any
import shutil

import cv2
from ai_edge_litert.interpreter import Interpreter
import numpy as np
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
    File,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from app_state import Board, app_state

PATH_LABEL = "models/labels.txt"
UPLOAD_DIR = Path("./models")

# Thread Control Flags & Locks
is_server_running = True
frame_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None

detection_control_event = threading.Event()
calibrate_control_event = threading.Event()

# PYDANTIC SCHEMAS


class ThreadInput(BaseModel):
    thread: int = 4


class CoreInput(BaseModel):
    core: List[int] = [0, 1, 2, 3]


class FpsInput(BaseModel):
    fps_camera: int = 5


class BoardCreate(BaseModel):
    board_id: str
    board_name: str
    ground_truth: List[str] = []


class BoardUpdate(BaseModel):
    board_name: Optional[str] = None
    ground_truth: Optional[List[str]] = None


class SelectModelRequest(BaseModel):
    model_name: str


# HELPER FUNCTIONS & CORE WORKER


def apply_core_affinity(core_list: List[int]) -> None:
    try:
        os.sched_setaffinity(0, set(core_list))
        print(
            f"📌 [CPU Manager] Thread successfully pinned to CPU Core(s): {core_list}"
        )
    except Exception as exc:
        print(f"❌ [CPU Manager] Failed to set CPU affinity: {exc}")

def detection() -> None:
    global latest_frame, is_server_running
    input_details = None
    output_details = None
    labels = {}
    
    try:
        with open(PATH_LABEL, "r") as f:
            for index, line in enumerate(f):
                labels[index] = line.strip()
    except FileNotFoundError:
        error_msg = f"Label file not found at '{PATH_LABEL}'"
        print(f"❌ [Detection Engine] {error_msg}")
        app_state.model.camera_error = error_msg
        return

    # Outer Loop: Survives throughout the application lifespan
    while is_server_running:
        is_det_active = detection_control_event.is_set()
        is_cal_active = calibrate_control_event.is_set()
        
        if not is_det_active and not is_cal_active:
            app_state.model.camera_error = None
            time.sleep(0.2)
            continue

        # --- Camera Hardware Initialization ---
        app_state.model.camera_error = None
        target_fps = getattr(app_state.model, "fps_camera", 15)

        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 660)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        if not cap.isOpened():
            error_msg = "Unable to open webcam device (/dev/video0)."
            print(f"❌ [Detection Engine] {error_msg}")
            app_state.model.camera_error = error_msg
            detection_control_event.clear()
            calibrate_control_event.clear()
            time.sleep(1.0) 
            continue

        # --- Proses Deteksi ---
        interpreter = None
        if is_det_active: 
            current_threads = app_state.model.thread
            active_cores = getattr(app_state.model, "core", [2, 3])
            apply_core_affinity(active_cores)

            try:
                interpreter = Interpreter(
                    model_path=Path("./models") / app_state.model.model,
                    num_threads=current_threads,
                )
                interpreter.allocate_tensors()
                app_state.model.need_reload = False
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
            except Exception as exc:
                error_msg = f"Failed to allocate TFLite interpreter: {exc}"
                print(f"❌ [Detection Engine] {error_msg}")
                app_state.model.camera_error = error_msg
                app_state.model.inference_fps = 0.0
                app_state.model.forward_pass_ms = 0.0
                cap.release()
                detection_control_event.clear()
                time.sleep(1.0)
                continue

        input_height = 320
        input_width = 320
        
        while is_server_running:
            current_det = detection_control_event.is_set()
            current_cal = calibrate_control_event.is_set()
            
            if not current_det and not current_cal or getattr(app_state.model, "need_reload", False):
                break

            start_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                error_msg = "Unable to open webcam device (/dev/video0)."
                app_state.model.camera_error = error_msg
                time.sleep(0.01)
                break
            
            # Mode Deteksi Aktif 
            if current_det and interpreter is not None and input_details is not None and output_details is not None:
                image_resized = cv2.resize(frame, (input_width, input_height))
                image_color = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(image_color, axis=0).astype(np.float32)
                input_data = input_data / 255.0
                
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                
                scores = interpreter.get_tensor(output_details[0]["index"])[0]
                boxes = interpreter.get_tensor(output_details[1]["index"])[0]
                num_detections = int(interpreter.get_tensor(output_details[2]["index"])[0])
                classes = interpreter.get_tensor(output_details[3]["index"])[0]
                
                # Draw Annotations
                for i in range(num_detections):
                    score = scores[i]
                    if score > 0.5:
                        ymin, xmin, ymax, xmax = boxes[i]
                        class_id = int(classes[i])
                        class_id += 1
                        
                        label = labels.get(class_id, f"ID {class_id}")

                        left = int(xmin * frame.shape[1])
                        right = int(xmax * frame.shape[1])
                        top = int(ymin * frame.shape[0])
                        bottom = int(ymax * frame.shape[0])

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{score*100:.1f}%",
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                end_frame = time.perf_counter()
                frame_duration = end_frame - start_frame
                fps = 1 / frame_duration if frame_duration > 0 else 0
                forward_pass = frame_duration * 1000

                app_state.model.inference_fps = round(fps, 2)
                app_state.model.forward_pass_ms = round(forward_pass, 2)
                
            else:
                app_state.model.inference_fps = 0.0
                app_state.model.forward_pass_ms = 0.0

            # Gambar hanya garis vertikal
            h_frame, w_frame = frame.shape[:2]
            w_box = 500
            x1 = (w_frame - w_box) // 2
            x2 = x1 + w_box
            cv2.line(frame, (x1, 0), (x1, h_frame), (0, 255, 255), 3)
            cv2.line(frame, (x2, 0), (x2, h_frame), (0, 255, 255), 3)
                
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.001)

        # Device cleanup upon exiting processing loop
        cap.release()
        app_state.model.inference_fps = 0.0
        app_state.model.forward_pass_ms = 0.0
        print("📸 [Detection Engine] Camera hardware handle released.")

    print("🛑 [Detection Engine] Worker thread terminated cleanly.")

async def generate_mjpeg_stream(request: Request):
    global latest_frame
    try:
        while True:
            if await request.is_disconnected():
                break

            img_to_encode = None
            with frame_lock:
                if latest_frame is not None:
                    img_to_encode = latest_frame.copy()

            if img_to_encode is None:
                await asyncio.sleep(0.01)
                continue

            ret, encoded_image = cv2.imencode(".jpg", img_to_encode)
            if not ret:
                await asyncio.sleep(0.01)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded_image.tobytes() + b"\r\n"
            )

            await asyncio.sleep(0.03)

    except asyncio.CancelledError:
        pass
    finally:
        with frame_lock:
            latest_frame = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global is_server_running
    is_server_running = True

    detection_thread = threading.Thread(target=detection, daemon=True)
    detection_thread.start()
    print("🚀 [System Init] Background detection thread started successfully.")

    yield

    print("⏳ [System Teardown] Shutting down services & background tasks...")
    is_server_running = False
    detection_control_event.clear()
    time.sleep(0.5)


app = FastAPI(title="EdgeLab-AI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ROUTERS DEFINITION
video_router = APIRouter(prefix="", tags=["Video Control"])
config_router = APIRouter(prefix="/api", tags=["Hardware & Model Configuration"])
file_router = APIRouter(prefix="/api", tags=["File Model Configuration"])
gt_router = APIRouter(prefix="/api/gt", tags=["Ground Truth Management"])
ws_router = APIRouter(prefix="/ws", tags=["WebSockets"])


@video_router.get("/start-detection")
async def start_detection():
    calibrate_control_event.clear() 
    detection_control_event.set()
    app_state.model.need_reload = True
    return {"status": "success", "message": "Camera stream with detection initiated."}


@video_router.get("/start-calibrate")
async def start_calibrate():
    detection_control_event.clear()
    calibrate_control_event.set()
    return {"status": "success", "message": "Camera stream for calibration initiated."}


@video_router.get("/stop")
async def stop_video():
    global latest_frame
    detection_control_event.clear()
    calibrate_control_event.clear()
    with frame_lock:
        latest_frame = None
    app_state.model.inference_fps = 0.0
    app_state.model.forward_pass_ms = 0.0
    return {
        "status": "success",
        "message": "Camera stream stopped. Hardware resource released.",
    }


@video_router.get("/video")
async def video_feed(request: Request):
    return StreamingResponse(
        generate_mjpeg_stream(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@config_router.get("/thread")
async def get_current_threads():
    return {"thread": app_state.model.thread}


@config_router.post("/thread")
async def set_thread_allocation(config: ThreadInput):
    try:
        if app_state.model.thread != config.thread:
            app_state.model.thread = config.thread
            app_state.model.need_reload = True
            return {"status": "success", "thread": app_state.model.thread}
        return {"status": "success", "message": "Thread count unchanged."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update thread allocation: {str(exc)}",
        )


@config_router.get("/core")
async def get_current_cores():
    core = getattr(app_state.model, "core", [2, 3])
    return {"core": core}


@config_router.post("/core")
async def set_core_allocation(config: CoreInput):
    try:
        current_cores = getattr(app_state.model, "core", [2, 3])
        if current_cores != config.core:
            app_state.model.core = config.core
            app_state.model.need_reload = True
            return {"status": "success", "core": app_state.model.core}
        return {"status": "success", "core": app_state.model.core}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update CPU core affinity: {str(exc)}",
        )


@config_router.get("/fps")
async def get_target_fps():
    try:
        fps_camera = getattr(app_state.model, "fps_camera", 5)
        return {"status": "success", "fps_camera": fps_camera}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve target FPS: {str(exc)}",
        )


@config_router.post("/fps")
async def set_target_fps(config: FpsInput):
    try:
        if config.fps_camera <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="FPS value must be greater than 0.",
            )
        current_fps = getattr(app_state.model, "fps_camera", 30)
        if current_fps != config.fps_camera:
            app_state.model.fps_camera = config.fps_camera
            app_state.model.need_reload = True
            return {"status": "success", "fps_camera": app_state.model.fps_camera}
        return {"status": "success", "message": "Target FPS unchanged."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update target FPS: {str(exc)}",
        )


@gt_router.get("")
async def get_all_boards():
    return {"status": "success", "data": app_state.gt_state.boards}


@gt_router.post("")
async def create_board(payload: BoardCreate):
    try:
        if any(b.board_id == payload.board_id for b in app_state.gt_state.boards):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Board ID '{payload.board_id}' already exists.",
            )

        new_board = Board(
            board_id=payload.board_id,
            board_name=payload.board_name,
            ground_truth=payload.ground_truth,
        )
        app_state.gt_state.boards.append(new_board)
        return {
            "status": "success",
            "message": "Board successfully created.",
            "data": new_board,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create board: {str(exc)}",
        )


@gt_router.put("/{board_id}")
async def update_board(board_id: str, payload: BoardUpdate):
    try:
        board = next(
            (b for b in app_state.gt_state.boards if b.board_id == board_id), None
        )
        if not board:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board with ID '{board_id}' was not found.",
            )

        if payload.board_name is not None:
            board.board_name = payload.board_name
        if payload.ground_truth is not None:
            board.ground_truth = payload.ground_truth

        return {
            "status": "success",
            "message": "Board successfully updated.",
            "data": board,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update board: {str(exc)}",
        )


@gt_router.delete("/{board_id}")
async def delete_board(board_id: str):
    try:
        idx = next(
            (
                i
                for i, b in enumerate(app_state.gt_state.boards)
                if b.board_id == board_id
            ),
            None,
        )
        if idx is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board with ID '{board_id}' was not found.",
            )

        deleted_board = app_state.gt_state.boards.pop(idx)
        return {
            "status": "success",
            "message": f"Board '{deleted_board.board_id}' deleted successfully.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete board: {str(exc)}",
        )


@file_router.post("/upload-models")
async def upload_file(file: UploadFile = File(...)):
    assert file.filename is not None
    filename = Path(file.filename).name
    file_location = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal menyimpan berkas: {str(e)}",
        )
    finally:
        await file.close()

    return {
        "status": "success",
        "filename": file.filename,
        "saved_path": file_location,
    }


@file_router.get("/models")
async def get_tflite_models():
    try:
        model_files = [
            f.name
            for f in UPLOAD_DIR.iterdir()
            if f.is_file() and f.name.endswith(".tflite")
        ]
        return {"selected_model": app_state.model.model, "models": model_files}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal membaca direktori: {str(e)}",
        )


@file_router.post("/models")
async def set_active_model(payload: SelectModelRequest):
    print(payload.model_name)
    selected_name = payload.model_name.strip()
    target_file = (UPLOAD_DIR / selected_name).with_suffix(".tflite")
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Berkas model '{selected_name}' tidak ditemukan di folder models/.",
        )
    app_state.model.model = selected_name
    app_state.model.need_reload = True
    return {
        "status": "success",
        "current_model": app_state.model.model,
    }


@ws_router.websocket("/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(
                {
                    "camera_error": getattr(app_state.model, "camera_error", None),
                    "inference_fps": getattr(app_state.model, "inference_fps", 0.0),
                    "forward_pass_ms": getattr(app_state.model, "forward_pass_ms", 0.0),
                }
            )
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        print("⚠️ [WebSocket] Client connection closed.")
    finally:
        print("🧹 [WebSocket] Connection cleanup completed.")


@config_router.get("/userspace-metrics")
async def get_userspace_metrics():
    try:
        fps_camera = getattr(app_state.model, "fps_camera")
        inference_fps = getattr(app_state.model, "inference_fps")
        return {
            "status": "success",
            "fps_camera": fps_camera,
            "inference_fps": inference_fps,
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data",
        )


app.include_router(video_router)
app.include_router(config_router)
app.include_router(file_router)
app.include_router(gt_router)
app.include_router(ws_router)