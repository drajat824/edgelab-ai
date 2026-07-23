import contextlib
import threading
import time
import os
import cv2
import asyncio
import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from ai_edge_litert.interpreter import Interpreter
from typing import List, Optional
from app_state import app_state, Board

# ==== VIDEO CONTROL ====
video_control_event = threading.Event()

# ==== LABEL ====

PATH_MODEL = "models/model.tflite"
PATH_LABEL = "models/labels.txt"

latest_frame = None
frame_lock = threading.Lock()

video_control_event = threading.Event()


# === INPUT ===


class ThreadInput(BaseModel):
    num_threads: int = 4


class CoreInput(BaseModel):
    cores: list[int] = [0, 1, 2, 3]


class FpsInput(BaseModel):
    fps_camera: int = 5


class BoardCreate(BaseModel):
    board_id: str
    board_name: str
    ground_truth: List[str] = []


class BoardUpdate(BaseModel):
    board_name: Optional[str] = None
    ground_truth: Optional[List[str]] = None


# === DETECTION ===


def detection():
    global latest_frame

    labels = {}
    try:
        with open(PATH_LABEL, "r") as f:
            for index, line in enumerate(f):
                labels[index] = line.strip()
    except FileNotFoundError:
        print(f"❌ Error: Label file not found at {PATH_LABEL}")
        app_state.model.camera_error = f"Label file not found at {PATH_LABEL}"
        return

    # 🟢 Outer Loop: Berjalan terus selama server hidup
    while True:
        # Jika sinyal stop aktif, tunggu di sini tanpa membunuh thread
        if not video_control_event.is_set():
            app_state.model.camera_error = None
            time.sleep(0.2)  # Jeda ramah CPU
            continue

        # --- Inisialisasi Kamera ---
        app_state.model.camera_error = None
        target_hardware_fps = getattr(app_state.model, "fps_camera", 15)

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 660)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
        cap.set(cv2.CAP_PROP_FPS, target_hardware_fps)

        if not cap.isOpened():
            print("❌ Error: Unable to open webcam!")
            app_state.model.camera_error = "Error: Unable to open webcam"
            video_control_event.clear()
            time.sleep(1.0)  # Beri waktu driver hardware melepaskan resource
            continue

        current_threads = app_state.model.num_threads
        active_cores = getattr(app_state.model, "cores", [2, 3])
        cores_config(active_cores)

        try:
            interpreter = Interpreter(
                model_path=PATH_MODEL, num_threads=current_threads
            )
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"❌ Failed to allocate TFLite interpreter: {e}")
            app_state.model.camera_error = f"Failed to allocate TFLite interpreter: {e}"
            cap.release()
            video_control_event.clear()
            time.sleep(1.0)
            continue

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_height = input_details[0]["shape"][1]
        input_width = input_details[0]["shape"][2]

        app_state.model.need_reload = False

        # 🟢 Inner Loop: Frame Processing
        while video_control_event.is_set():
            if getattr(app_state.model, "need_reload", False):
                print("⚠️ Reloading thread/core configuration...")
                break

            start_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue  # Jangan break! Biarkan mencoba membaca frame berikutnya

            # Process TFLite inference
            image_resized = cv2.resize(frame, (input_width, input_height))
            input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[0]["index"])[0]
            classes = interpreter.get_tensor(output_details[1]["index"])[0]
            scores = interpreter.get_tensor(output_details[2]["index"])[0]
            num_detections = int(interpreter.get_tensor(output_details[3]["index"])[0])

            for i in range(num_detections):
                score = scores[i]
                if score > 0.5:
                    ymin, xmin, ymax, xmax = boxes[i]
                    class_id = int(classes[i])
                    label = labels.get(class_id, f"ID {class_id}")
                    left = int(xmin * frame.shape[1])
                    right = int(xmax * frame.shape[1])
                    top = int(ymin * frame.shape[0])
                    bottom = int(ymax * frame.shape[0])
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} ({score*100:.1f}%)",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            end_frame = time.perf_counter()
            fps = 1 / (end_frame - start_frame) if (end_frame - start_frame) > 0 else 0
            forward_pass = (end_frame - start_frame) * 1000

            app_state.model.inference_fps = round(fps, 2)
            app_state.model.forward_pass_ms = round(forward_pass, 2)

            # Draw board bounding area
            tinggi_frame, lebar_frame, _ = frame.shape
            lebar_kotak, tinggi_kotak = 400, 300
            x1 = (lebar_frame - lebar_kotak) // 2
            y1 = (tinggi_frame - tinggi_kotak) // 2
            cv2.rectangle(
                frame, (x1, y1), (x1 + lebar_kotak, y1 + tinggi_kotak), (0, 255, 255), 3
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Forward-pass Time: {forward_pass:.2f} ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            with frame_lock:
                latest_frame = frame.copy()

            time.sleep(0.001)

        # Clean up kamera saat keluar dari inner loop
        cap.release()
        print("📸 Camera device released.")

    print("🛑 Detection thread safely terminated.")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    
    detection_thread = threading.Thread(target=detection, daemon=True)
    detection_thread.start()
    print("🚀 Background Thread Deteksi Berhasil Dijalankan.")

    yield

    print("⏳ Server mematikan koneksi & background tasks...")
    video_control_event.clear()
    time.sleep(0.5)


app = FastAPI(title="EdgeLab-AI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate(request: Request):
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

            frame_bytes = encoded_image.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            await asyncio.sleep(0.03)

    except asyncio.CancelledError:
        pass
    finally:
        with frame_lock:
            latest_frame = None


# VIDEO


@app.get("/start")
async def start_video():
    video_control_event.set()
    return {"status": "success", "message": "Sinyal start kamera dikirim."}


@app.get("/stop")
async def stop_video():
    global latest_frame
    video_control_event.clear()
    with frame_lock:
        latest_frame = None
    return {
        "status": "success",
        "message": "Sinyal stop kamera dikirim. Hardware dilepas.",
    }


@app.get("/video")
async def video_feed(request: Request):
    video_control_event.set()
    return StreamingResponse(
        generate(request), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# THREAD & CORE


@app.post("/api/thread")
async def handle_thread_state(config: ThreadInput):
    try:
        data = config.num_threads
        if app_state.model.num_threads != data:
            app_state.model.num_threads = data
            app_state.model.need_reload = True
            return {"status": "success", "num_threads": app_state.model.num_threads}
        return {"status": "success", "message": "Thread tidak berubah."}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update thread allocation: {str(e)}",
        )


@app.get("/api/thread")
async def get_current_threads():
    return {"num_threads": app_state.model.num_threads}


# CORES


def cores_config(core_list: list[int]):
    try:
        # Mengunci thread yang sedang berjalan saat ini ke daftar core yang dipilih
        os.sched_setaffinity(0, set(core_list))
        print(f"📌 [AI Server] Thread berhasil dikunci ke Core CPU: {core_list}")
    except Exception as e:
        print(f"❌ Gagal mengunci Core CPU: {e}")


@app.post("/api/cores")
async def handle_cores_state(config: CoreInput):
    try:
        data_cores = config.cores
        current_cores = getattr(app_state.model, "cores", [2, 3])
        if current_cores != data_cores:
            app_state.model.cores = data_cores
            app_state.model.need_reload = True

            return {"status": "success", "cores": app_state.model.cores}

        return {"status": "success", "cores": app_state.model.cores}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch cores allocation: {str(e)}"
        )


@app.get("/api/cores")
async def get_current_cores():
    cores = getattr(app_state.model, "cores", [2, 3])
    return {"cores": cores}


# FPS CAMERA


@app.get("/api/fps")
async def get_current_fps():
    try:
        fps_camera = getattr(app_state.model, "fps_camera", 5)

        return {"status": "success", "fps_camera": fps_camera}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch FPS metrics: {str(e)}"
        )


@app.post("/api/fps")
async def handle_fps_state(config: FpsInput):
    try:
        data_fps = config.fps_camera
        if data_fps <= 0:
            raise HTTPException(status_code=400, detail="FPS harus lebih besar dari 0")
        current_fps_target = getattr(app_state.model, "fps_camera", 30)
        if current_fps_target != data_fps:
            app_state.model.fps_camera = data_fps
            app_state.model.need_reload = True
            return {"status": "success", "fps_camera": app_state.model.fps_camera}
        return {"status": "success", "message": "Target FPS tidak berubah."}
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update FPS target: {str(e)}"
        )


# GET GT
@app.get("/api/gt")
async def get_all_boards():
    return {"status": "success", "data": app_state.gt_state.boards}


# CREATE GT
@app.post("/api/gt")
async def create_board(payload: BoardCreate):
    try:
        # Check duplicate board_id
        if any(b.board_id == payload.board_id for b in app_state.gt_state.boards):
            raise HTTPException(status_code=400, detail="Board ID already exists")

        new_board = Board(
            board_id=payload.board_id,
            board_name=payload.board_name,
            ground_truth=payload.ground_truth,
        )
        app_state.gt_state.boards.append(new_board)

        return {"status": "success", "message": "Board created", "data": new_board}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create board: {str(e)}")


# UPDATE GT
@app.put("/api/gt/{board_id}")
async def update_board(board_id: str, payload: BoardUpdate):
    try:
        board = next(
            (b for b in app_state.gt_state.boards if b.board_id == board_id), None
        )
        if not board:
            raise HTTPException(status_code=404, detail="Board not found")

        # Update properties if provided
        if payload.board_name is not None:
            board.board_name = payload.board_name
        if payload.ground_truth is not None:
            board.ground_truth = payload.ground_truth

        return {"status": "success", "message": "Board updated", "data": board}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update board: {str(e)}")


# DELETE GT
@app.delete("/api/gt/{board_id}")
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
            raise HTTPException(status_code=404, detail="Board not found")

        deleted_board = app_state.gt_state.boards.pop(idx)
        return {
            "status": "success",
            "message": f"Board {deleted_board.board_id} deleted",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete board: {str(e)}")


# WEB SOCKET


router = APIRouter()


@router.websocket("/ws/inference")
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
        print("⚠️ Client WebSocket disconnected.")
    finally:
        print("🧹 Cleanup WebSocket Done.")


app.include_router(router)
