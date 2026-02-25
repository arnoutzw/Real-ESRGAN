"""Tkinter GUI for Real-ESRGAN video upscaling with drag-and-drop support."""

import cv2
import mimetypes
import numpy as np
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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


def get_upsampler(model_name, denoise_strength, tile, outscale):
    """Create and return a RealESRGANer instance."""
    model, netscale, urls = build_model(model_name)

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
    return upsampler


def is_video_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return True
    mime = mimetypes.guess_type(path)[0]
    return mime is not None and mime.startswith('video')


class VideoUpscaleApp:

    def __init__(self, root):
        self.root = root
        self.root.title('Real-ESRGAN Video Upscaler')
        self.root.geometry('700x620')
        self.root.resizable(True, True)
        self.root.minsize(600, 550)

        self.video_path = None
        self.output_path = None
        self.is_processing = False

        self._build_ui()
        self._setup_drag_and_drop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Style
        style = ttk.Style()
        style.configure('Drop.TFrame', relief='ridge')
        style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))

        pad = dict(padx=10, pady=5)

        # --- Device info ---
        device = get_device()
        device_labels = {'cuda': 'NVIDIA GPU (CUDA)', 'mps': 'Apple Silicon (MPS)', 'cpu': 'CPU'}
        device_text = device_labels.get(device.type, str(device))

        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill='x', **pad)
        ttk.Label(info_frame, text='Real-ESRGAN Video Upscaler', style='Header.TLabel').pack(side='left')
        ttk.Label(info_frame, text=f'Device: {device_text}').pack(side='right')

        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10)

        # --- Drop zone ---
        self.drop_frame = tk.Frame(
            self.root, bg='#e8e8e8', relief='groove', bd=2, cursor='hand2')
        self.drop_frame.pack(fill='both', expand=True, padx=15, pady=10)

        self.drop_label = tk.Label(
            self.drop_frame,
            text='Drop a video file here\n\nor click to browse',
            font=('Helvetica', 14),
            bg='#e8e8e8', fg='#555555',
            justify='center',
        )
        self.drop_label.pack(expand=True, fill='both')

        # Click to browse
        self.drop_frame.bind('<Button-1>', lambda e: self._browse_file())
        self.drop_label.bind('<Button-1>', lambda e: self._browse_file())

        # --- Video info ---
        self.video_info_var = tk.StringVar(value='No video selected')
        ttk.Label(self.root, textvariable=self.video_info_var, wraplength=650).pack(**pad)

        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10)

        # --- Settings ---
        settings_frame = ttk.Frame(self.root)
        settings_frame.pack(fill='x', **pad)

        # Model
        ttk.Label(settings_frame, text='Model:').grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.model_var = tk.StringVar(value='realesr-animevideov3')
        model_combo = ttk.Combobox(
            settings_frame, textvariable=self.model_var,
            values=MODEL_OPTIONS, state='readonly', width=30)
        model_combo.grid(row=0, column=1, sticky='w', padx=(0, 20))

        # Scale
        ttk.Label(settings_frame, text='Scale:').grid(row=0, column=2, sticky='w', padx=(0, 5))
        self.scale_var = tk.IntVar(value=2)
        scale_spin = ttk.Spinbox(
            settings_frame, from_=1, to=4, textvariable=self.scale_var, width=5)
        scale_spin.grid(row=0, column=3, sticky='w')

        # Tile size
        ttk.Label(settings_frame, text='Tile size:').grid(row=1, column=0, sticky='w', padx=(0, 5), pady=(5, 0))
        self.tile_var = tk.IntVar(value=0)
        tile_spin = ttk.Spinbox(
            settings_frame, from_=0, to=800, increment=32, textvariable=self.tile_var, width=8)
        tile_spin.grid(row=1, column=1, sticky='w', pady=(5, 0))
        ttk.Label(settings_frame, text='(0 = no tiling)').grid(row=1, column=2, columnspan=2, sticky='w', pady=(5, 0))

        # Denoise strength
        ttk.Label(settings_frame, text='Denoise:').grid(row=2, column=0, sticky='w', padx=(0, 5), pady=(5, 0))
        self.denoise_var = tk.DoubleVar(value=0.5)
        denoise_scale = ttk.Scale(
            settings_frame, from_=0, to=1, variable=self.denoise_var, orient='horizontal', length=180)
        denoise_scale.grid(row=2, column=1, sticky='w', pady=(5, 0))
        ttk.Label(settings_frame, text='(general-x4v3 only)').grid(
            row=2, column=2, columnspan=2, sticky='w', pady=(5, 0))

        # --- Output path ---
        out_frame = ttk.Frame(self.root)
        out_frame.pack(fill='x', **pad)
        ttk.Label(out_frame, text='Output folder:').pack(side='left')
        self.output_dir_var = tk.StringVar(value=os.path.join(ROOT_DIR, 'results'))
        ttk.Entry(out_frame, textvariable=self.output_dir_var, width=45).pack(side='left', padx=5)
        ttk.Button(out_frame, text='Browse', command=self._browse_output_dir).pack(side='left')

        # --- Progress ---
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x', padx=15, pady=(5, 2))

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(self.root, textvariable=self.status_var).pack(**pad)

        # --- Buttons ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=15, pady=(0, 10))

        self.upscale_btn = ttk.Button(
            btn_frame, text='Upscale Video', command=self._start_upscale)
        self.upscale_btn.pack(side='right', padx=5)

        self.open_output_btn = ttk.Button(
            btn_frame, text='Open Output Folder', command=self._open_output_folder, state='disabled')
        self.open_output_btn.pack(side='right', padx=5)

    # ------------------------------------------------------------------
    # Drag-and-drop support
    # ------------------------------------------------------------------

    def _setup_drag_and_drop(self):
        """Try to enable native drag-and-drop via tkinterdnd2."""
        try:
            from tkinterdnd2 import DND_FILES
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self._on_drop)
            self.drop_frame.dnd_bind('<<DragEnter>>', self._on_drag_enter)
            self.drop_frame.dnd_bind('<<DragLeave>>', self._on_drag_leave)
        except ImportError:
            # tkinterdnd2 not installed — click-to-browse still works
            pass

    def _on_drag_enter(self, event):
        self.drop_frame.configure(bg='#c8daf0')
        self.drop_label.configure(bg='#c8daf0')

    def _on_drag_leave(self, event):
        self.drop_frame.configure(bg='#e8e8e8')
        self.drop_label.configure(bg='#e8e8e8')

    def _on_drop(self, event):
        self.drop_frame.configure(bg='#e8e8e8')
        self.drop_label.configure(bg='#e8e8e8')
        # tkinterdnd2 wraps paths with spaces in braces: {/path/to file}
        path = event.data.strip()
        if path.startswith('{') and path.endswith('}'):
            path = path[1:-1]
        # Handle multiple files — take the first one
        if '\n' in path:
            path = path.split('\n')[0].strip()
        self._load_video(path)

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _browse_file(self):
        if self.is_processing:
            return
        path = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=[
                ('Video files', '*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.webm *.m4v *.mpg *.mpeg'),
                ('All files', '*.*'),
            ])
        if path:
            self._load_video(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title='Select output folder')
        if path:
            self.output_dir_var.set(path)

    def _load_video(self, path):
        if self.is_processing:
            return

        if not os.path.isfile(path):
            messagebox.showerror('Error', f'File not found:\n{path}')
            return

        if not is_video_file(path):
            messagebox.showerror('Error', 'The selected file does not appear to be a video.')
            return

        self.video_path = path

        # Read video metadata
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror('Error', 'Could not open video file.')
            self.video_path = None
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = total_frames / fps if fps > 0 else 0
        fname = os.path.basename(path)

        self.video_info_var.set(
            f'{fname}  |  {w}x{h}  |  {fps:.1f} fps  |  {total_frames} frames  |  {duration:.1f}s')

        self.drop_label.configure(
            text=f'{fname}\n{w}x{h} @ {fps:.1f} fps\n\nClick to change',
            fg='#222222')

        self.progress_var.set(0)
        self.status_var.set('Ready — press Upscale Video to start')

    # ------------------------------------------------------------------
    # Upscaling
    # ------------------------------------------------------------------

    def _start_upscale(self):
        if self.is_processing:
            return
        if self.video_path is None:
            messagebox.showwarning('No video', 'Please select or drop a video file first.')
            return

        self.is_processing = True
        self.upscale_btn.configure(state='disabled')
        self.open_output_btn.configure(state='disabled')
        self.progress_var.set(0)
        self.status_var.set('Starting...')

        thread = threading.Thread(target=self._upscale_worker, daemon=True)
        thread.start()

    def _upscale_worker(self):
        try:
            self._run_upscale()
        except Exception as exc:
            self.root.after(0, lambda: self._on_error(str(exc)))

    def _run_upscale(self):
        model_name = self.model_var.get()
        outscale = self.scale_var.get()
        tile = self.tile_var.get()
        denoise = self.denoise_var.get()
        output_dir = self.output_dir_var.get()

        os.makedirs(output_dir, exist_ok=True)

        # Status: loading model
        self.root.after(0, lambda: self.status_var.set(f'Loading model {model_name}...'))

        upsampler = get_upsampler(model_name, denoise, tile, outscale)

        # Open input video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = int(w * outscale), int(h * outscale)

        # Write upscaled frames to a temp file
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
                output, _ = upsampler.enhance(frame, outscale=outscale)
            except RuntimeError as e:
                # Fallback: bicubic resize
                output = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            writer.write(output)
            frame_idx += 1

            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                self.root.after(0, lambda p=pct, f=frame_idx, t=total_frames:
                                self._update_progress(p, f, t))

        cap.release()
        writer.release()

        # Build output filename
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        final_output = os.path.join(output_dir, f'{base_name}_{model_name}_x{outscale}.mp4')

        # Re-encode with ffmpeg and copy audio
        self.root.after(0, lambda: self.status_var.set('Encoding final video...'))
        cmd = [
            'ffmpeg', '-y',
            '-i', tmp_video_noaudio,
            '-i', self.video_path,
            '-map', '0:v',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'copy',
            '-shortest',
            '-loglevel', 'error',
            final_output,
        ]
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ffmpeg failed or missing — use the raw OpenCV output
            import shutil
            shutil.move(tmp_video_noaudio, final_output)

        # Cleanup temp dir
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        self.output_path = final_output
        self.root.after(0, self._on_complete)

    def _update_progress(self, pct, frame, total):
        self.progress_var.set(pct)
        self.status_var.set(f'Upscaling frame {frame}/{total} ({pct:.1f}%)')

    def _on_complete(self):
        self.is_processing = False
        self.progress_var.set(100)
        self.status_var.set(f'Done! Saved to: {self.output_path}')
        self.upscale_btn.configure(state='normal')
        self.open_output_btn.configure(state='normal')
        messagebox.showinfo('Complete', f'Video saved to:\n{self.output_path}')

    def _on_error(self, msg):
        self.is_processing = False
        self.progress_var.set(0)
        self.status_var.set(f'Error: {msg}')
        self.upscale_btn.configure(state='normal')
        messagebox.showerror('Error', f'Upscaling failed:\n{msg}')

    def _open_output_folder(self):
        folder = self.output_dir_var.get()
        if not os.path.isdir(folder):
            return
        if sys.platform == 'darwin':
            subprocess.Popen(['open', folder])
        elif sys.platform == 'win32':
            os.startfile(folder)
        else:
            subprocess.Popen(['xdg-open', folder])


def main():
    # Try to use tkinterdnd2's TkinterDnD.Tk for native drag-and-drop
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()

    VideoUpscaleApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
