import asyncio
import contextlib
import logging
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app_state import Board, app_state

logger = logging.getLogger(__name__)

PATH_LABEL = "models/labels.txt"
UPLOAD_DIR = Path("./models")

# Thread Control Flags & Locks
is_server_running = True

# --- Arsitektur Frame Baru ---
raw_frame: Optional[np.ndarray] = None
raw_frame_lock = threading.Lock()

latest_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()
# -----------------------------

detection_control_event = threading.Event()
calibrate_control_event = threading.Event()
camera_hardware = 2

is_detection_running_now = False

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

class Point(BaseModel):
    x: float
    y: float

class StartDetection(BaseModel):
    calibration_points: List[Point] = []

class ActiveBoard(BaseModel):
    active_board: str


#  ==== START SORTED CARDS ====

def get_board_matrices(
    calibration_points: List[Any], target_w: int = 3900, target_h: int = 3180
) -> Tuple[np.ndarray, np.ndarray]:
    pts_src = np.array(
        [
            [p.x, p.y] if hasattr(p, "x") else [p["x"], p["y"]] if isinstance(p, dict) else p
            for p in calibration_points
        ],
        dtype=np.float32,
    )

    pts_dst = np.array(
        [
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)

    return M, M_inv


def warp_board(image: np.ndarray, M: np.ndarray, target_w: int = 320, target_h: int = 320) -> np.ndarray:
    return cv2.warpPerspective(image, M, (target_w, target_h))


def get_slot_center(img_w: int = 3900, img_h: int = 3180) -> Dict[int, Tuple[float, float]]:
    scale_x = img_w / 39.0
    scale_y = img_h / 31.8

    margin_x_px = 2.0 * scale_x
    margin_y_px = 2.0 * scale_y
    gap_x_px = 1.0 * scale_x
    gap_y_px = 1.0 * scale_y

    slot_w_px = (img_w - (2 * margin_x_px) - (4 * gap_x_px)) / 5
    slot_h_px = (img_h - (2 * margin_y_px) - (2 * gap_y_px)) / 3

    slot_centers = {}
    for row in range(3):
        for col in range(5):
            slot_id = (row * 5) + col + 1
            center_x = margin_x_px + (col * (slot_w_px + gap_x_px)) + (slot_w_px / 2.0)
            center_y = margin_y_px + (row * (slot_h_px + gap_y_px)) + (slot_h_px / 2.0)
            slot_centers[slot_id] = (center_x, center_y)

    return slot_centers


def map_to_15_slots(
    detections: List[Dict[str, Any]], slot_centers: Dict[int, Tuple[float, float]]
) -> List[Dict[str, Any]]:
    grid_slots = {
        sid: {
            "slot_id": sid,
            "label": "NULL",
            "confidence": 0.0,
        }
        for sid in range(1, 16)
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        best_slot_id = None
        min_distance = float("inf")

        for slot_id, (scx, scy) in slot_centers.items():
            dist = math.sqrt((cx - scx) ** 2 + (cy - scy) ** 2)
            if dist < min_distance:
                min_distance = dist
                best_slot_id = slot_id

        if best_slot_id is not None:
            current = grid_slots[best_slot_id]
            if current["label"] == "NULL" or det["confidence"] > current["confidence"]:
                grid_slots[best_slot_id] = {
                    "slot_id": best_slot_id,
                    "label": det["label"],
                    "confidence": round(float(det["confidence"]), 2),
                }

    return [grid_slots[i] for i in range(1, 16)]


# ==== METRICS ====
def match_boards(final_15_slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    global is_detection_running_now

    active_board = getattr(app_state.gt_state, "active_board", None)
    is_active_none = not active_board or str(active_board).upper() == "NONE"

    gt_slots: list[str | None] = []
    total_gt_cards = 0

    if not is_active_none:
        boards = app_state.gt_state.boards
        current_board = next((b for b in boards if b.board_id == active_board), None)
        if not current_board:
            raise ValueError(f"Board dengan ID '{active_board}' tidak ditemukan.")
        
        gt_slots = list(current_board.ground_truth)
        total_gt_cards = len([gt for gt in gt_slots if gt not in (None, "")])

    slot_details = []
    detected_slots = []

    for idx, det in enumerate(final_15_slots):
        slot_num = idx + 1
        label = det.get("label") if isinstance(det, dict) else None
        confidence = float(det.get("confidence", 0.0)) if isinstance(det, dict) else 0.0
        is_valid_det = det and label not in (None, "", "NULL")

        if is_active_none:
            gt = None
            is_correct = None
        else:
            gt = gt_slots[idx] if idx < len(gt_slots) else None
            is_correct = (label == gt) if is_valid_det else False

        detail = {
            "slot": slot_num,
            "detection": label,
            "ground_truth": gt,
            "confidence": confidence,
            "is_correct": is_correct,
        }
        slot_details.append(detail)

        if is_valid_det:
            detected_slots.append(detail)

    total_detections = len(detected_slots)
    avg_confidence = (
        sum(item["confidence"] for item in detected_slots) / total_detections
        if total_detections > 0
        else 0.0
    )

    if is_active_none:
        detection_rate = None
        precision = None
    else:
        detection_rate = (total_detections / total_gt_cards * 100) if total_gt_cards > 0 else 0.0
        if total_detections > 0:
            correct_detections = sum(1 for item in detected_slots if item["is_correct"])
            precision = (correct_detections / total_detections * 100)
        else:
            precision = 0.0

    result = {
        "metrics": {
            "detection_rate": detection_rate,
            "avg_confidence": avg_confidence,
            "precision": precision,
        },
        "slot_details": slot_details,
    }

    app_state.model.latest_evaluation = result
    return result


# HELPER FUNCTIONS & CORE WORKER
def apply_core_affinity(core_list: List[int]) -> None:
    try:
        thread_id = threading.get_native_id()
        os.sched_setaffinity(thread_id, set(core_list))
    except Exception as exc:
        print(f"❌ [CPU Manager] Failed to set CPU affinity: {exc}")


# ==== 1. THREAD KAMERA (PRODUCER) ====
def camera_worker() -> None:
    global raw_frame, is_server_running
    
    cap = None
    failed_read_count = 0  # Tambahkan counter deteksi kegagalan
    
    while is_server_running:
        if not detection_control_event.is_set() and not calibrate_control_event.is_set():
            if cap is not None and cap.isOpened():
                cap.release()
                cap = None
            time.sleep(0.1)
            continue
            
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(camera_hardware, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 660)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
                fps = getattr(app_state.model, "fps_camera", 15)
                cap.set(cv2.CAP_PROP_FPS, fps)
                failed_read_count = 0 # Reset counter jika berhasil buka
            else:
                app_state.model.camera_error = "Unable to open webcam device."
                time.sleep(2.0) # Jeda lebih lama di Pi jika gagal buka
                continue

        ret, frame = cap.read()
        if ret:
            with raw_frame_lock:
                raw_frame = frame.copy()
            failed_read_count = 0 # Reset counter jika berhasil baca
        else:
            failed_read_count += 1
            app_state.model.camera_error = f"Failed to capture frame ({failed_read_count}/5)"
            
            # Jika gagal baca 5x berturut-turut, paksa reset kamera
            if failed_read_count >= 5:
                print("⚠️ [Hardware] Kamera macet, mencoba re-inisialisasi...")
                cap.release()
                cap = None
                time.sleep(1.0)
            
        time.sleep(0.005)

    if cap is not None and cap.isOpened():
        cap.release()

# ==== 2. THREAD DETEKSI (CONSUMER 1) ====
def detection() -> None:
    global latest_frame, is_server_running, is_detection_running_now, raw_frame
    app_state.model.camera_error = None
    slot_centers = get_slot_center()

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

    while is_server_running:
        if not detection_control_event.is_set():
            time.sleep(0.1)
            continue
        
        is_detection_running_now = True
        app_state.model.camera_error = None
        calibration_points = getattr(app_state.model, "calibration_points", [])

        current_threads = app_state.model.thread
        active_cores = getattr(app_state.model, "core", [2, 3])
        apply_core_affinity(active_cores)

        try:
            print(f"Loading Model: {app_state.model.model}")
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
            app_state.model.camera_error = error_msg
            app_state.model.inference_fps = 0.0
            app_state.model.forward_pass_ms = 0.0
            detection_control_event.clear()
            time.sleep(1.0)
            continue

        M_320, _ = get_board_matrices(calibration_points, target_w=320, target_h=320)
        _, M_inv = get_board_matrices(calibration_points, target_w=3900, target_h=3180)

        while is_server_running and detection_control_event.is_set():
            if getattr(app_state.model, "need_reload", False):
                app_state.model.need_reload = False
                break

            start_frame = time.perf_counter()
            
            frame = None
            with raw_frame_lock:
                if raw_frame is not None:
                    frame = raw_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue

            warped_board = warp_board(frame, M_320, target_w=320, target_h=320)
            if warped_board is None or not isinstance(warped_board, np.ndarray):
                continue

            if interpreter is not None and input_details is not None and output_details is not None:
                try:
                    image_color = cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB)
                    input_data = np.expand_dims(image_color, axis=0).astype(np.float32) / 255.0

                    interpreter.set_tensor(input_details[0]["index"], input_data)
                    interpreter.invoke()

                    scores = interpreter.get_tensor(output_details[0]["index"])[0]
                    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
                    num_detections = int(interpreter.get_tensor(output_details[2]["index"])[0])
                    classes = interpreter.get_tensor(output_details[3]["index"])[0]
                except Exception as exc:
                    error_msg = f"Failed inference execution: {exc}"
                    app_state.model.camera_error = error_msg
                    time.sleep(0.01)
                    break

                raw_detections = []
                frame_h, frame_w = frame.shape[:2]

                for i in range(num_detections):
                    score = scores[i]
                    if score > 0.5:
                        ymin, xmin, ymax, xmax = boxes[i]
                        class_id = int(classes[i]) + 1
                        label = labels.get(class_id, f"ID {class_id}")

                        left_b = xmin * 3900
                        right_b = xmax * 3900
                        top_b = ymin * 3180
                        bottom_b = ymax * 3180

                        raw_detections.append(
                            {
                                "label": label,
                                "confidence": float(score),
                                "bbox": [int(left_b), int(top_b), int(right_b), int(bottom_b)],
                            }
                        )

                        bbox_pts = np.array(
                            [
                                [[left_b, top_b]],
                                [[right_b, top_b]],
                                [[right_b, bottom_b]],
                                [[left_b, bottom_b]],
                            ],
                            dtype=np.float32,
                        )

                        transformed_pts = cv2.perspectiveTransform(bbox_pts, M_inv)
                        x_coords = transformed_pts[:, 0, 0]
                        y_coords = transformed_pts[:, 0, 1]

                        left_raw = max(0, int(np.min(x_coords)))
                        top_raw = max(0, int(np.min(y_coords)))
                        right_raw = min(frame_w, int(np.max(x_coords)))
                        bottom_raw = min(frame_h, int(np.max(y_coords)))

                        cv2.rectangle(
                            frame, (left_raw, top_raw), (right_raw, bottom_raw), (0, 255, 0), 2
                        )

                final_15_slots = map_to_15_slots(raw_detections, slot_centers)
                match_boards(final_15_slots)

                end_frame = time.perf_counter()
                frame_duration = end_frame - start_frame
                fps = 1 / frame_duration if frame_duration > 0 else 0
                forward_pass = frame_duration * 1000

                app_state.model.inference_fps = round(fps, 2)
                app_state.model.forward_pass_ms = round(forward_pass, 2)

            with frame_lock:
                latest_frame = frame.copy()

            time.sleep(0.001)

        is_detection_running_now = False
        time.sleep(0.1)


# ==== 3. THREAD KALIBRASI (CONSUMER 2) ====
def calibration() -> None:
    global latest_frame, is_server_running, raw_frame

    while is_server_running:
        if not calibrate_control_event.is_set():
            time.sleep(0.1)
            continue

        app_state.model.camera_error = None

        while is_server_running and calibrate_control_event.is_set():
            frame = None
            with raw_frame_lock:
                if raw_frame is not None:
                    frame = raw_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue

            h_frame, w_frame = frame.shape[:2]
            w_box = 500
            x1 = (w_frame - w_box) // 2
            x2 = x1 + w_box

            cv2.line(frame, (x1, 0), (x1, h_frame), (0, 255, 255), 3)
            cv2.line(frame, (x2, 0), (x2, h_frame), (0, 255, 255), 3)

            with frame_lock:
                latest_frame = frame.copy()
                
            time.sleep(0.01)


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

    camera_thread = threading.Thread(target=camera_worker, daemon=True, name="CameraWorker")
    detection_thread = threading.Thread(target=detection, daemon=True, name="DetectionWorker")
    calibration_thread = threading.Thread(target=calibration, daemon=True, name="CalibrationWorker")

    camera_thread.start()
    detection_thread.start()
    calibration_thread.start()

    print("🚀 [System Init] Camera, Detection & Calibration workers started.")

    try:
        yield
    finally:
        is_server_running = False
        detection_control_event.clear()
        calibrate_control_event.clear()
        time.sleep(0.5)


app = FastAPI(title="EdgeLab-AI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://edgelab.local:5173", "http://localhost:5173"],
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


@video_router.post("/start-detection")
async def start_detection(config: StartDetection):
    global latest_frame
    latest_frame = None
    app_state.model.camera_error = None
    app_state.model.latest_evaluation = None
    app_state.model.inference_fps = 0.0
    app_state.model.forward_pass_ms = 0.0
    app_state.model.calibration_points = config.calibration_points
    app_state.model.need_reload = True
    
    # Matikan event kalibrasi terlebih dahulu
    calibrate_control_event.clear()
    
    # Nyalakan event deteksi (instan tanpa time.sleep keras)
    detection_control_event.set()    
    return {"status": "success", "message": "Camera stream with detection initiated."}


@video_router.get("/start-calibrate")
async def start_calibrate():
    global latest_frame
    latest_frame = None
    app_state.model.latest_evaluation = None
    app_state.model.inference_fps = 0.0
    app_state.model.forward_pass_ms = 0.0
    
    # Matikan event deteksi terlebih dahulu
    detection_control_event.clear()
    
    # Nyalakan event kalibrasi
    calibrate_control_event.set()
    return {"status": "success", "message": "Camera stream for calibration initiated."}


@video_router.get("/stop")
async def stop_video():
    global latest_frame
    detection_control_event.clear()
    calibrate_control_event.clear()
    with frame_lock:
        latest_frame = None
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


@gt_router.post("/active-board")
async def set_active_board(payload: ActiveBoard):
    try:
        if payload.active_board is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'active_board' tidak boleh null.",
            )

        app_state.gt_state.active_board = payload.active_board
        return {
            "status": "success",
            "message": "Active boards unchanged.",
            "data": payload.active_board,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@gt_router.get("/active-board")
async def get_active_board():
    try:
        return {
            "status": "success",
            "message": "Active board retrieved successfully.",
            "data": app_state.gt_state.active_board,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
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
    selected_name = payload.model_name.strip()
    target_file = (UPLOAD_DIR / selected_name).with_suffix(".tflite")
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Berkas model '{selected_name}' tidak ditemukan di folder models/.",
        )
    app_state.model.model = selected_name
    return {
        "status": "success",
        "current_model": app_state.model.model,
    }


@ws_router.websocket("/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 [WebSocket] Client connected.")

    try:
        while True:
            model_state = getattr(app_state, "model", None)
            evaluation_data = getattr(app_state.model, "latest_evaluation", None)
            payload = {
                "camera_error": getattr(model_state, "camera_error", None) if model_state else None,
                "inference_fps": getattr(model_state, "inference_fps", 0.0) if model_state else 0.0,
                "forward_pass_ms": getattr(model_state, "forward_pass_ms", 0.0) if model_state else 0.0,
            }

            if evaluation_data is not None:
                payload["evaluation"] = evaluation_data

            await websocket.send_json(payload)
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("⚠️ [WebSocket] Client disconnected gracefully.")
    except RuntimeError:
        logger.warning("⚠️ [WebSocket] Runtime error (connection lost during send).")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("🧹 [WebSocket] Connection cleanup completed.")


@config_router.get("/userspace-metrics")
async def get_userspace_metrics():
    global is_detection_running_now

    try:
        fps_camera = getattr(app_state.model, "fps_camera", 5)
        inference_fps = getattr(app_state.model, "inference_fps", 0.0)
        return {
            "status": "success",
            "fps_camera": fps_camera,
            "inference_fps": inference_fps,
            "detection_run": is_detection_running_now,
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve data",
        )


app.include_router(video_router)
app.include_router(config_router)
app.include_router(file_router)
app.include_router(gt_router)
app.include_router(ws_router)