import contextlib
import threading
import time
import os
import cv2
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from ai_edge_litert.interpreter import Interpreter
from app_state import app_state

# ==== VIDEO CONTROL ====
video_control_event = threading.Event()

# ==== LABEL ====

PATH_MODEL = "models/model.tflite"
PATH_LABEL = "models/labels.txt"

latest_frame = None
frame_lock = threading.Lock()

# === DETECTION ===


def detection():
    global latest_frame

    labels = {}
    try:
        with open(PATH_LABEL, "r") as f:
            for index, line in enumerate(f):
                labels[index] = line.strip()
    except FileNotFoundError:
        print(f"❌ Error: File label tidak ditemukan di {PATH_LABEL}")
        return

    # Loop utama thread background
    while True:
        if not video_control_event.is_set():
            time.sleep(0.1)
            continue

        target_hardware_fps = getattr(app_state.model, "fps_camera", app_state.model.num_threads)

        # JIKA SAKLAR MENYALA: Buka hardware kamera
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 660)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
        cap.set(cv2.CAP_PROP_FPS, target_hardware_fps)

        real_hardware_fps = cap.get(cv2.CAP_PROP_FPS)
        print(
            f"📸 [Hardware] Anda meminta: {target_hardware_fps} FPS | Driver V4L2 memberikan: {real_hardware_fps} FPS"
        )

        if not cap.isOpened():
            print("❌ Error: Tidak bisa membuka webcam!")
            video_control_event.clear()
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
            print(f"❌ Gagal alokasi interpreter: {e}")
            cap.release()
            time.sleep(1)
            continue

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_height = input_details[0]["shape"][1]
        input_width = input_details[0]["shape"][2]

        app_state.model.need_reload = False

        # Loop pemrosesan frame
        while True:
            # INTERUPSI 1: Jika tombol /stop ditekan (video_control_event dimatikan)
            if not video_control_event.is_set():
                print("🛑 [AI Server] Tombol stop ditekan. Mematikan kamera...")
                break

            # INTERUPSI 2: Jika ada sinyal ganti thread
            if getattr(app_state.model, "need_reload", False):
                print("⚠️ [AI Server] Mendapat sinyal perubahan thread. Reloading...")
                break

            start_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # --- (Sisa kode inferensi TFLite Anda ke bawah tetap sama) ---
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
            fps = 1 / (end_frame - start_frame)
            forward_pass = (end_frame - start_frame) * 1000

            # Batas Papan
            tinggi_frame, lebar_frame, _ = frame.shape
            lebar_kotak = 400
            tinggi_kotak = 300
            x1 = (lebar_frame - lebar_kotak) // 2
            y1 = (tinggi_frame - tinggi_kotak) // 2
            x2 = x1 + lebar_kotak
            y2 = y1 + tinggi_kotak
            titik_kiri_atas = (x1, y1)
            titik_kanan_bawah = (x2, y2)
            warna = (0, 255, 255)
            ketebalan = 3
            cv2.rectangle(frame, titik_kiri_atas, titik_kanan_bawah, warna, ketebalan)

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

        # Keluar dari loop frame -> Lepas hardware kamera dengan aman
        cap.release()
        print("🔌 [AI Server] Hardware kamera berhasil dilepas secara aman.")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    detection_thread = threading.Thread(target=detection, daemon=True)
    detection_thread.start()
    print("🚀 Background Thread Deteksi Berhasil Dijalankan.")
    yield


app = FastAPI(title="EdgeLab-AI API", lifespan=lifespan)

# === MIDDLEWARE CORS ===
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
        video_control_event.clear()
        with frame_lock:
            latest_frame = None

# === CORE CONFIG ===

def cores_config(core_list: list[int]):
    try:
        # Mengunci thread yang sedang berjalan saat ini ke daftar core yang dipilih
        os.sched_setaffinity(0, set(core_list))
        print(f"📌 [AI Server] Thread berhasil dikunci ke Core CPU: {core_list}")
    except Exception as e:
        print(f"❌ Gagal mengunci Core CPU: {e}")


# === INPUT ===

class ThreadInput(BaseModel):
    num_threads: int = 4


class CoreInput(BaseModel):
    cores: list[int] = [0, 1, 2, 3]


class FpsInput(BaseModel):
    fps_camera: int = 5


# === API - BACKEND ===

# START - STOP

@app.get("/start")
async def start_video():
    if video_control_event.is_set():
        return {"status": "info", "message": "Kamera sudah dalam kondisi menyala."}
    video_control_event.set()
    return {"status": "success", "message": "Sinyal start kamera dikirim."}


@app.get("/stop")
async def stop_video():
    global latest_frame
    if not video_control_event.is_set():
        return {"status": "info", "message": "Kamera sudah dalam kondisi mati."}
    video_control_event.clear()
    with frame_lock:
        latest_frame = None
    return {
        "status": "success",
        "message": "Sinyal stop kamera dikirim. Hardware dilepas.",
    }


# THREAD


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


# STREAMING


@app.get("/video")
async def video_feed(request: Request):
    video_control_event.set()

    return StreamingResponse(
        generate(request), media_type="multipart/x-mixed-replace; boundary=frame"
    )
