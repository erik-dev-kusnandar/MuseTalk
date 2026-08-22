"""
MuseTalk FastAPI Service - AI Seminar Integration (V2 - Optimized)
Berjalan di port 8003 (diubah untuk testing) dengan musetalk_env venv.
Endpoint: POST /generate-video → return video MP4
"""

import os
import sys
import time
import uuid
import shutil
import queue
import asyncio
import threading
import argparse
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2  # Dipindahkan ke atas
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# MuseTalk imports
from omegaconf import OmegaConf
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Konfigurasi ───────────────────────────────────────────────
MUSETALK_DIR   = Path(__file__).parent.resolve()
AVATAR_ID      = os.getenv("MUSETALK_AVATAR_ID", "avator_1")
AVATAR_VERSION = "v15"
MODEL_UNET     = str(MUSETALK_DIR / "models/musetalkV15/unet.pth")
MODEL_CONFIG   = str(MUSETALK_DIR / "models/musetalkV15/musetalk.json")
WHISPER_DIR    = str(MUSETALK_DIR / "models/whisper")

# ─── App ───────────────────────────────────────────────────────
app = FastAPI(title="MuseTalk API Service - Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model state ────────────────────────────────────────
device       = None
vae          = None
unet         = None
pe           = None
whisper      = None
audio_proc   = None
fp           = None
timesteps    = None
weight_dtype = None
avatar_cache = {}

# ─── Avatar class ──────────────────────────────────────────────
class Avatar:
    # OPTIMASI: Batch size default dinaikkan ke 24 untuk efisiensi VRAM 12GB
    def __init__(self, avatar_id: str, batch_size: int = 24):
        self.avatar_id   = avatar_id
        self.base_path   = str(MUSETALK_DIR / f"results/{AVATAR_VERSION}/avatars/{avatar_id}")
        self.full_imgs_path    = f"{self.base_path}/full_imgs"
        self.coords_path       = f"{self.base_path}/coords.pkl"
        self.latents_out_path  = f"{self.base_path}/latents.pt"
        self.video_out_path    = f"{self.base_path}/vid_output/"
        self.mask_out_path     = f"{self.base_path}/mask"
        self.mask_coords_path  = f"{self.base_path}/mask_coords.pkl"
        self.batch_size        = batch_size
        self.idx = 0
        os.makedirs(self.video_out_path, exist_ok=True)
        self._load_prepared()

    def _load_prepared(self):
        import pickle, glob
        logger.info(f"Loading prepared avatar: {self.avatar_id}")

        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        self.frame_list_cycle = read_imgs(input_img_list)

        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)

        input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')))
        self.mask_list_cycle = read_imgs(input_mask_list)
        logger.info(f"Avatar loaded: {len(self.frame_list_cycle)} frames")

    # OPTIMASI: Menggunakan FFmpeg subprocess Pipe, tanpa cv2.imwrite
    def process_frames(self, res_frame_queue, video_num, ffmpeg_proc):
        processed_count = 0
        while processed_count < video_num:
            if res_frame_queue.empty():
                time.sleep(0.005) # Sleep diperkecil
                continue
                
            res_frame = res_frame_queue.get(block=False)
            bbox     = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                self.idx += 1
                processed_count += 1
                continue
                
            ori_frame = self.frame_list_cycle[self.idx % len(self.frame_list_cycle)].copy()
            mask      = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            
            combine   = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop)
            
            # Tulis frame array langsung ke memori ffmpeg (pipe)
            try:
                ffmpeg_proc.stdin.write(combine.tobytes())
            except BrokenPipeError:
                logger.error("FFmpeg pipe broken!")
                break
                
            self.idx += 1
            processed_count += 1
            
        # Tutup stdin agar ffmpeg tau video sudah selesai
        if ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    @torch.no_grad()
    def inference(self, audio_path: str, out_vid_name: str, fps: int = 25) -> str:
        self.idx = 0

        whisper_input_features, librosa_length = audio_proc.get_audio_feature(
            audio_path, weight_dtype=weight_dtype
        )
        whisper_chunks = audio_proc.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper,
            librosa_length, fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        video_num = len(whisper_chunks)
        
        # Ambil dimensi frame asli untuk pipe ffmpeg
        if len(self.frame_list_cycle) > 0:
            frame_h, frame_w, _ = self.frame_list_cycle[0].shape
        else:
            frame_h, frame_w = 720, 1280 # Fallback
            
        temp_mp4 = f"{self.base_path}/temp_{uuid.uuid4().hex}.mp4"
        
        # OPTIMASI: Jalankan ffmpeg di background menerima raw video dari pipe (stdin)
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{frame_w}x{frame_h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', # Input dari stdin
            '-c:v', 'libx264', '-preset', 'faster', '-pix_fmt', 'yuv420p', 
            '-crf', '23', temp_mp4
        ]
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        res_frame_queue = queue.Queue()
        process_thread = threading.Thread(
            target=self.process_frames,
            args=(res_frame_queue, video_num, ffmpeg_proc)
        )
        process_thread.start()

        # Proses AI UNet
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        for whisper_batch, latent_batch in gen:
            audio_feat = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
            pred = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
            pred = pred.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred)
            for frame in recon:
                res_frame_queue.put(frame)

        # Tunggu thread blending selesai
        process_thread.join()

        # OPTIMASI: Gabungkan audio dan temp video
        output_vid = os.path.join(self.video_out_path, f"{out_vid_name}.mp4")
        os.system(
            f"ffmpeg -y -v warning -i {temp_mp4} -i {audio_path} "
            f"-c:v copy -c:a aac -shortest {output_vid}"
        )

        # Cleanup
        if os.path.exists(temp_mp4):
            os.remove(temp_mp4)

        logger.info(f"Video saved: {output_vid}")
        return output_vid

# ─── Startup: load models ──────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global device, vae, unet, pe, whisper, audio_proc, fp, timesteps, weight_dtype

    logger.info("Loading MuseTalk models...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae, unet, pe = load_all_model(
        unet_model_path=MODEL_UNET,
        vae_type="sd-vae",
        unet_config=MODEL_CONFIG,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    # OPTIMASI: Channels last untuk memory footprint efisien di arsitektur Ampere/Turing
    unet.model = unet.model.to(memory_format=torch.channels_last)

    pe       = pe.half().to(device)
    vae.vae  = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    weight_dtype = unet.model.dtype

    audio_proc = AudioProcessor(feature_extractor_path=WHISPER_DIR)
    whisper = WhisperModel.from_pretrained(WHISPER_DIR)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

    logger.info(f"Pre-loading default avatar: {AVATAR_ID}")
    try:
        # OPTIMASI: Default batch size 24
        avatar_cache[AVATAR_ID] = Avatar(avatar_id=AVATAR_ID, batch_size=8)
    except Exception as e:
        logger.warning(f"Could not load default avatar {AVATAR_ID}: {e}")
        
    logger.info("MuseTalk API V2 ready! ✅")

# ─── Avatar preparation (in-process) ──────────────────────────
prep_lock = asyncio.Lock()


def _prepare_avatar_material(video_path: str, avatar_id: str) -> None:
    """In-process port of realtime_inference.Avatar.prepare_material (v15).

    Reuses models already loaded in this process (vae, fp, dwpose/face-detect
    via musetalk.utils.preprocessing) instead of spawning a subprocess that
    duplicates every model and blows up VRAM.
    """
    import glob as glob_mod
    import pickle

    base_path = str(MUSETALK_DIR / f"results/{AVATAR_VERSION}/avatars/{avatar_id}")
    full_imgs_path = f"{base_path}/full_imgs"
    mask_out_path = f"{base_path}/mask"
    coords_path = f"{base_path}/coords.pkl"
    latents_out_path = f"{base_path}/latents.pt"
    mask_coords_path = f"{base_path}/mask_coords.pkl"

    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(full_imgs_path, exist_ok=True)
    os.makedirs(mask_out_path, exist_ok=True)

    logger.info(f"[{avatar_id}] extracting video frames...")
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{full_imgs_path}/{count:08d}.png", frame)
        count += 1
    cap.release()

    input_img_list = sorted(glob_mod.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]")))
    if not input_img_list:
        raise RuntimeError("No frames could be extracted from the reference video")

    logger.info(f"[{avatar_id}] extracting landmarks from {len(input_img_list)} frames...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)

    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    extra_margin = 10
    for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if tuple(bbox) == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + extra_margin, frame.shape[0])
        coord_list[idx] = [x1, y1, x2, y2]
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(vae.get_latents_for_unet(resized_crop_frame))

    if not input_latent_list:
        raise RuntimeError("No face detected in the reference video")

    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    latent_list_cycle = input_latent_list + input_latent_list[::-1]

    logger.info(f"[{avatar_id}] building face masks ({len(frame_list_cycle)} frames)...")
    mask_coords_list_cycle = []
    for i, frame in enumerate(frame_list_cycle):
        cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png", frame)
        x1, y1, x2, y2 = coord_list_cycle[i]
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode="jaw")
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
        mask_coords_list_cycle.append(crop_box)

    with open(mask_coords_path, "wb") as fobj:
        pickle.dump(mask_coords_list_cycle, fobj)
    with open(coords_path, "wb") as fobj:
        pickle.dump(coord_list_cycle, fobj)
    torch.save(latent_list_cycle, latents_out_path)
    logger.info(f"[{avatar_id}] avatar material ready")


# ─── Endpoints ────────────────────────────────────────────────
@app.post("/prepare-avatar")
async def prepare_avatar(file: UploadFile = File(...)):
    avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
    file_ext = Path(file.filename).suffix.lower()

    tmp_upload = tempfile.mktemp(suffix=file_ext)
    with open(tmp_upload, "wb") as f:
        f.write(await file.read())

    try:
        video_path = str(MUSETALK_DIR / f"data/video/{avatar_id}.mp4")
        if file_ext in [".jpg", ".jpeg", ".png"]:
            cmd = f"ffmpeg -y -loop 1 -i {tmp_upload} -c:v libx264 -t 2 -pix_fmt yuv420p -r 25 {video_path}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(tmp_upload, video_path)

        async with prep_lock:
            await asyncio.to_thread(_prepare_avatar_material, video_path, avatar_id)
            avatar_cache[avatar_id] = Avatar(avatar_id=avatar_id, batch_size=8)

        return {"status": "success", "avatar_id": avatar_id}
    except Exception as e:
        logger.exception(f"[{avatar_id}] prepare-avatar failed")
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_upload):
            os.remove(tmp_upload)

@app.get("/health")
async def health():
    return {"status": "ready" if avatar_cache else "loading", "device": str(device)}

inference_lock = asyncio.Lock()

@app.post("/generate-video")
async def generate_video(audio_file: UploadFile = File(...), avatar_id: Optional[str] = None):
    target_id = avatar_id or AVATAR_ID
    
    if target_id not in avatar_cache:
        try:
            avatar_cache[target_id] = Avatar(avatar_id=target_id, batch_size=8)
        except Exception as e:
            raise HTTPException(404, f"Avatar not found: {e}")

    obj = avatar_cache[target_id]
    suffix = Path(audio_file.filename).suffix or ".wav"
    tmp_audio = tempfile.mktemp(suffix=suffix)
    with open(tmp_audio, "wb") as f:
        f.write(await audio_file.read())

    try:
        out_name = f"ai_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        
        async with inference_lock:
            output_path = await asyncio.to_thread(obj.inference, tmp_audio, out_name, fps=25)
            
        logger.info(f"Video generated in {time.time()-t0:.1f}s")
        return FileResponse(output_path, media_type="video/mp4", filename=f"{out_name}.mp4")
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)

if __name__ == "__main__":
    os.chdir(MUSETALK_DIR)
    # Jalankan di port 8003 agar tidak bentrok dengan yang sedang jalan
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")