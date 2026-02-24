"""Gradio web UI for Real-ESRGAN — drag-and-drop image and video upscaling."""

import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import tempfile

import gradio as gr
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_OPTIONS = [
    'RealESRGAN_x4plus',
    'RealESRNet_x4plus',
    'RealESRGAN_x4plus_anime_6B',
    'RealESRGAN_x2plus',
    'realesr-animevideov3',
    'realesr-general-x4v3',
]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(model_name):
    """Return (network, netscale, file_urls) for a given model name."""
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        urls = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ]
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model, netscale, urls


def get_upsampler(model_name, denoise_strength, tile, face_enhance, outscale):
    """Create and return a RealESRGANer (and optional face enhancer)."""
    model, netscale, urls = build_model(model_name)

    # Resolve model path, downloading if needed
    model_path = os.path.join(ROOT_DIR, 'weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in urls:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    device = get_device()
    # MPS does not support half precision reliably
    half = device.type == 'cuda'

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )

    face_enhancer_obj = None
    if face_enhance:
        try:
            from gfpgan import GFPGANer
            face_enhancer_obj = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        except ImportError:
            print('GFPGAN not installed, skipping face enhancement.')

    return upsampler, face_enhancer_obj


# ---------------------------------------------------------------------------
# Image upscaling
# ---------------------------------------------------------------------------

def upscale_image(input_image, model_name, outscale, denoise_strength, tile, face_enhance):
    if input_image is None:
        raise gr.Error('Please upload an image.')

    outscale = int(outscale)
    tile = int(tile)

    upsampler, face_enhancer_obj = get_upsampler(
        model_name, denoise_strength, tile, face_enhance, outscale)

    # Gradio gives us a numpy RGB array
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    try:
        if face_enhancer_obj is not None:
            _, _, output = face_enhancer_obj.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise gr.Error(f'Inference failed: {e}. Try a smaller tile size.')

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


# ---------------------------------------------------------------------------
# Video upscaling
# ---------------------------------------------------------------------------

def upscale_video(input_video, model_name, outscale, denoise_strength, tile, face_enhance, progress=gr.Progress()):
    if input_video is None:
        raise gr.Error('Please upload a video.')

    outscale = int(outscale)
    tile = int(tile)

    upsampler, face_enhancer_obj = get_upsampler(
        model_name, denoise_strength, tile, face_enhance, outscale)

    # Read video with OpenCV
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise gr.Error('Could not open video file.')

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = int(w * outscale), int(h * outscale)

    # Write upscaled frames to a temp file (without audio)
    tmp_dir = tempfile.mkdtemp()
    tmp_video_noaudio = os.path.join(tmp_dir, 'noaudio.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_video_noaudio, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if face_enhancer_obj is not None:
                _, _, output = face_enhancer_obj.enhance(
                    frame, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(frame, outscale=outscale)
        except RuntimeError as e:
            print(f'Frame {frame_idx} error: {e}')
            output = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        writer.write(output)
        frame_idx += 1
        if total_frames > 0:
            progress(frame_idx / total_frames, desc=f'Upscaling frame {frame_idx}/{total_frames}')

    cap.release()
    writer.release()

    # Re-encode with ffmpeg for better compatibility and copy audio from source
    output_path = os.path.join(tmp_dir, 'output.mp4')
    cmd = [
        'ffmpeg', '-y',
        '-i', tmp_video_noaudio,
        '-i', input_video,
        '-map', '0:v',
        '-map', '1:a?',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        '-shortest',
        '-loglevel', 'error',
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg unavailable or failed — fall back to the raw file
        output_path = tmp_video_noaudio

    return output_path


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def create_app():
    device = get_device()
    device_label = {'cuda': 'NVIDIA GPU (CUDA)', 'mps': 'Apple Silicon GPU (MPS)', 'cpu': 'CPU'}
    detected = device_label.get(device.type, str(device))

    with gr.Blocks(title='Real-ESRGAN Upscaler', theme=gr.themes.Soft()) as app:
        gr.Markdown(f'# Real-ESRGAN Upscaler\nDrag and drop an image or video to upscale it.  '
                    f'Detected device: **{detected}**')

        with gr.Tabs():
            # ---- Image tab ----
            with gr.TabItem('Image'):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type='numpy', label='Input Image')
                        with gr.Row():
                            img_model = gr.Dropdown(
                                choices=MODEL_OPTIONS, value='RealESRGAN_x4plus', label='Model')
                            img_scale = gr.Slider(
                                minimum=1, maximum=8, value=4, step=1, label='Output Scale')
                        with gr.Row():
                            img_denoise = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.1,
                                label='Denoise Strength (general-x4v3 only)')
                            img_tile = gr.Slider(
                                minimum=0, maximum=800, value=0, step=32,
                                label='Tile Size (0 = no tiling)')
                        img_face = gr.Checkbox(label='Face Enhancement (GFPGAN)', value=False)
                        img_btn = gr.Button('Upscale Image', variant='primary')
                    with gr.Column():
                        img_output = gr.Image(type='numpy', label='Output Image')

                img_btn.click(
                    fn=upscale_image,
                    inputs=[img_input, img_model, img_scale, img_denoise, img_tile, img_face],
                    outputs=img_output,
                )

            # ---- Video tab ----
            with gr.TabItem('Video'):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label='Input Video')
                        with gr.Row():
                            vid_model = gr.Dropdown(
                                choices=MODEL_OPTIONS, value='realesr-animevideov3', label='Model')
                            vid_scale = gr.Slider(
                                minimum=1, maximum=4, value=2, step=1, label='Output Scale')
                        with gr.Row():
                            vid_denoise = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.1,
                                label='Denoise Strength (general-x4v3 only)')
                            vid_tile = gr.Slider(
                                minimum=0, maximum=800, value=0, step=32,
                                label='Tile Size (0 = no tiling)')
                        vid_face = gr.Checkbox(label='Face Enhancement (GFPGAN)', value=False)
                        vid_btn = gr.Button('Upscale Video', variant='primary')
                    with gr.Column():
                        vid_output = gr.Video(label='Output Video')

                vid_btn.click(
                    fn=upscale_video,
                    inputs=[vid_input, vid_model, vid_scale, vid_denoise, vid_tile, vid_face],
                    outputs=vid_output,
                )

    return app


if __name__ == '__main__':
    app = create_app()
    app.launch()
