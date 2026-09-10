import os
from dotenv import load_dotenv
import sys
import uuid
import base64
import urllib.request
import numpy as np
import cv2
import datetime
import asyncpg
import asyncio
import gc
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse

if int(cv2.__version__.split(".")[0]) >= 5:
    raise RuntimeError("OpenCV 5.0+ dropped Caffe model support. Downgrade your environment.")

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

load_dotenv() 

# This will raise KeyError if DATABASE_URL is not set
database_url = os.environ['DATABASE_URL']

DB_URL = os.getenv("DATABASE_URL", database_url)
db_pool = None
ai_task_queue = asyncio.Queue()
workers = []
ai_semaphore = asyncio.Semaphore(1)

FACE_PROTO, FACE_MODEL = "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
GENDER_PROTO, GENDER_MODEL = "gender_deploy.prototxt", "gender_net.caffemodel"
AGE_PROTO, AGE_MODEL = "age_deploy.prototxt", "age_net.caffemodel"

MODEL_URLS = {
    FACE_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    FACE_MODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    GENDER_PROTO: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_deploy.prototxt",
    GENDER_MODEL: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_net.caffemodel",
    AGE_PROTO: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/age_deploy.prototxt",
    AGE_MODEL: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/age_net.caffemodel"
}

for file_name, url in MODEL_URLS.items():
    if not os.path.exists(file_name):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_name, 'wb') as out_file:
            out_file.write(response.read())

# Load models ONCE globally to fix memory duplication limit
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
for net in [face_net, gender_net, age_net]:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def decode_base64_image(base64_str: str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        np_arr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def analyze_frame(img, user_gender):
    if img is None: return False, False, False, "unknown"
    try:
        global face_net, gender_net, age_net
        h, w = img.shape[:2]
        
        blob_face = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        face_net.setInput(blob_face)
        detections = face_net.forward()
        
        face_found = False
        best_box = None
        max_conf = 0
        
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.55 and conf > max_conf: # Slightly stricter face confidence
                max_conf = conf
                best_box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                face_found = True
        
        is_kid = False
        predicted_gender = "unknown"
        is_nudity = False
        x1 = y1 = x2 = y2 = 0
        
        if face_found:
            startX, startY, endX, endY = best_box
            pad_x = int((endX - startX) * 0.15)
            pad_y = int((endY - startY) * 0.20)
            x1, y1 = max(0, startX - pad_x), max(0, startY - pad_y)
            x2, y2 = min(w, endX + pad_x), min(h, endY + pad_y)
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.4, 87.8, 114.9), swapRB=False)
                
                # AGE DETECTION (Optimized)
                age_net.setInput(blob)
                age_preds = age_net.forward()[0]
                # Brackets 0-3 cover ages 0 to 20. Require 70% confidence to flag as minor.
                minor_prob = float(np.sum(age_preds[0:4]))
                is_kid = bool(minor_prob > 0.70)
                
                # GENDER DETECTION (Optimized)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()[0]
                male_conf = float(gender_preds[0])
                female_conf = float(gender_preds[1])
                
                if male_conf > 0.65:
                    predicted_gender = "male"
                elif female_conf > 0.65:
                    predicted_gender = "female"
                else:
                    predicted_gender = user_gender
            
        # NUDITY DETECTION (Optimized using YCrCb & Torso Isolation)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # YCrCb is highly resistant to lighting changes compared to HSV
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
        
        if face_found:
            # Black out the face AND neck area to prevent false positives
            mask[max(0, y1-20):y2, x1:x2] = 0 
        
        # Only scan the lower 2/3rds of the screen (chest/torso area)
        torso_mask = mask[int(h*0.33):h, 0:w]
        if torso_mask.size > 0:
            skin_ratio = np.sum(torso_mask > 0) / torso_mask.size
            is_nudity = bool(skin_ratio > 0.45) # If 45% of torso area is bare skin, flag nudity
        
        return face_found, is_kid, is_nudity, predicted_gender
    except Exception:
        return False, False, False, "unknown"

async def ai_background_worker():
    loop = asyncio.get_running_loop()
    while True:
        try:
            task = await ai_task_queue.get()
            client_ip, ws, user_gender, room_id, img = task
            
            if ws.client_state.name != "CONNECTED":
                ai_task_queue.task_done()
                continue

            async with ai_semaphore:
                face_found, is_kid, is_nudity, predicted_gender = await loop.run_in_executor(None, analyze_frame, img, user_gender)
                del img
                gc.collect()
            
            status_text = "No Face"
            if is_nudity:
                status_text = "Nudity Detected"
            elif face_found:
                status_text = predicted_gender.capitalize()
                if is_kid:
                    status_text += " (Minor)"

            try:
                await ws.send_json({"type": "ai_status", "payload": f"AI: {status_text}"})
            except Exception:
                pass

            if is_nudity:
                await execute_ban(client_ip, ws, room_id, "Explicit/Nudity content detected.")
            elif is_kid:
                await execute_ban(client_ip, ws, room_id, "Minors are strictly prohibited.")
            elif not face_found:
                try: await ws.send_json({"type": "warning", "payload": "⚠️ Warning: Face not visible! Please stay in the camera view."})
                except Exception: pass
            elif predicted_gender != user_gender and predicted_gender != "unknown":
                try: await ws.send_json({"type": "gender_mismatch", "payload": "⚠️ Warning: Detected gender does not match your selection."})
                except Exception: pass

            ai_task_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception:
            ai_task_queue.task_done()

async def execute_ban(client_ip, ws, room_id, reason):
    await ban_user(client_ip, reason)
    try:
        await ws.send_json({"type": "error", "payload": f"⛔ Banned: {reason}"})
        await asyncio.sleep(0.2)
        await ws.close()
    except Exception: pass

    for chat_ws, ip in list(client_ips.items()):
        if ip == client_ip:
            try:
                await chat_ws.send_json({"type": "error", "payload": f"⛔ Banned: {reason}"})
                r_id = user_rooms.get(chat_ws)
                if r_id and r_id in active_rooms:
                    for client in active_rooms[r_id]:
                        if client != chat_ws:
                            await client.send_json({"type": "system", "payload": "Stranger was banned for safety violations."})
                            await client.send_json({"type": "peer_disconnected"})
                            await asyncio.sleep(0.2)
                            await client.close()
                    active_rooms.pop(r_id, None)
                await asyncio.sleep(0.2)
                await chat_ws.close()
            except Exception: pass

@app.on_event("startup")
async def startup():
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DB_URL, statement_cache_size=0, max_inactive_connection_lifetime=300)
        async with db_pool.acquire() as conn:
            await conn.execute('CREATE TABLE IF NOT EXISTS banned_ips (ip VARCHAR(255) PRIMARY KEY, reason TEXT, is_banned BOOLEAN DEFAULT TRUE)')
            await conn.execute('CREATE TABLE IF NOT EXISTS ads (id SERIAL PRIMARY KEY, ad_content TEXT, is_active BOOLEAN DEFAULT TRUE)')
    except Exception: pass
    for _ in range(4): workers.append(asyncio.create_task(ai_background_worker()))

@app.on_event("shutdown")
async def shutdown():
    if db_pool: await db_pool.close()
    for worker in workers: worker.cancel()

async def is_banned(ip: str):
    if not db_pool: return None
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT reason FROM banned_ips WHERE ip = $1 AND is_banned = TRUE", ip)
            return row["reason"] if row else None
    except Exception as e:
        print(f"DB Ban Check Error: {e}")
        return None

async def ban_user(ip: str, reason: str):
    if not db_pool: return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO banned_ips (ip, reason, is_banned) 
                VALUES ($1, $2, TRUE) 
                ON CONFLICT (ip) DO UPDATE SET is_banned = TRUE, reason = EXCLUDED.reason
            """, ip, reason)
    except Exception as e: 
        print(f"DB Ban Insert Error: {e}")

waiting_males, waiting_females = [], []
active_rooms, user_rooms, client_ips = {}, {}, {}
SCAM_WORDS = ["crypto", "invest", "cashapp", "venmo", "telegram", "whatsapp", "paypal", "bitcoin", "scam", "hack"]

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

@app.get("/api/ad")
async def get_ad():
    return {"ad_content": "<div style='color:#fff;'>[ Default Ad Banner ]</div>"}

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    client_ip = websocket.client.host
    try:
        await websocket.accept()
        if await is_banned(client_ip):
            await websocket.close()
            return
        while True:
            data = await websocket.receive_json()
            img = decode_base64_image(data.get("image", ""))
            user_gender = data.get("gender", "male").lower()
            current_room = next((user_rooms.get(ws) for ws, ip in client_ips.items() if ip == client_ip), None)
            await ai_task_queue.put((client_ip, websocket, user_gender, current_room, img))
    except Exception: pass

@app.websocket("/ws/chat/{gender}")
async def websocket_chat(websocket: WebSocket, gender: str):
    client_ip = websocket.client.host
    loop = asyncio.get_running_loop()
    try:
        await websocket.accept()
        ban_reason = await is_banned(client_ip)
        if ban_reason:
            await websocket.send_json({"type": "error", "payload": f"⛔ Banned: {ban_reason}"})
            await websocket.close()
            return

        client_ips[websocket] = client_ip
        user_gender = gender.lower()
        is_verified = False

        init_data = await websocket.receive_json()
        if init_data.get("type") == "verify":
            if not init_data.get("policy_accepted"):
                await websocket.send_json({"type": "error", "payload": "❌ Must accept policies."})
                await websocket.close()
                return
            
            img = decode_base64_image(init_data.get("image", ""))
            
            async with ai_semaphore:
                face_found, is_kid, is_nudity, predicted_gender = await loop.run_in_executor(None, analyze_frame, img, user_gender)
                del img
                gc.collect()

            if not face_found:
                await websocket.send_json({"type": "error", "payload": "❌ No human face detected."})
                await websocket.close()
                return

            if is_kid or is_nudity:
                await ban_user(client_ip, "Policy violation detected on connection.")
                await websocket.send_json({"type": "error", "payload": "⛔ Banned: Policy violation."})
                await websocket.close()
                return

            if predicted_gender != user_gender and predicted_gender != "unknown":
                await websocket.send_json({"type": "gender_mismatch", "payload": "⚠️ Warning: Detected gender does not match selection."})

            is_verified = True
            await websocket.send_json({"type": "status", "payload": "✅ Verified. Joining matchmaking..."})

        if not is_verified: return

        if user_gender == "male":
            if waiting_females:
                partner_ws = waiting_females.pop(0)
                room_id = str(uuid.uuid4())
                active_rooms[room_id] = [websocket, partner_ws]
                user_rooms[websocket] = user_rooms[partner_ws] = room_id
                await websocket.send_json({"type": "match_start", "role": "initiator", "partner_gender": "Female"})
                await partner_ws.send_json({"type": "match_start", "role": "receiver", "partner_gender": "Male"})
            else:
                waiting_males.append(websocket)
                await websocket.send_json({"type": "status", "payload": "Searching for a user..."})
        else:
            if waiting_males:
                partner_ws = waiting_males.pop(0)
                room_id = str(uuid.uuid4())
                active_rooms[room_id] = [partner_ws, websocket]
                user_rooms[websocket] = user_rooms[partner_ws] = room_id
                await partner_ws.send_json({"type": "match_start", "role": "initiator", "partner_gender": "Female"})
                await websocket.send_json({"type": "match_start", "role": "receiver", "partner_gender": "Male"})
            else:
                waiting_females.append(websocket)
                await websocket.send_json({"type": "status", "payload": "Searching for a male user..."})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            current_room = user_rooms.get(websocket)
            if not current_room or current_room not in active_rooms: continue

            if msg_type in ["offer", "answer", "candidate"]:
                for client in active_rooms[current_room]:
                    if client != websocket: await client.send_json(data)
            elif msg_type == "text":
                text_payload = data.get("payload", "").lower()
                if any(word in text_payload for word in SCAM_WORDS):
                    await ban_user(client_ip, "Scam/Spam violations.")
                    await websocket.send_json({"type": "error", "payload": "⛔ Banned for Scam violations."})
                    for client in active_rooms[current_room]:
                        if client != websocket:
                            await client.send_json({"type": "system", "payload": "Stranger banned for scamming."})
                            await client.send_json({"type": "peer_disconnected"})
                            await client.close()
                    await websocket.close()
                    break
                for client in active_rooms[current_room]:
                    if client != websocket: await client.send_json({"type": "message", "payload": data.get("payload")})
            elif msg_type == "report":
                for client in active_rooms[current_room]:
                    if client != websocket:
                        await ban_user(client_ips.get(client), "Reported by user.")
                        await client.send_json({"type": "error", "payload": "⛔ You were reported and banned."})
                        await websocket.send_json({"type": "system", "payload": "User banned."})
                        await client.close()
                await websocket.send_json({"type": "peer_disconnected"})
                break
    except Exception:
        if websocket in waiting_males: waiting_males.remove(websocket)
        if websocket in waiting_females: waiting_females.remove(websocket)
        client_ips.pop(websocket, None)
        current_room = user_rooms.pop(websocket, None)
        if current_room and current_room in active_rooms:
            partners = active_rooms.pop(current_room)
            for client in partners:
                if client != websocket:
                    user_rooms.pop(client, None)
                    try:
                        await client.send_json({"type": "peer_disconnected"})
                        await client.close()
                    except Exception: pass
