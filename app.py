import os
import sys
import socket

# Đảm bảo mã hóa UTF-8 trên Windows console và tắt cảnh báo symlinks
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import numpy as np
import soundfile as sf
import gradio as gr
from vieneu import Vieneu

# ── Khởi tạo mô hình VieNeu-TTS (ONNX Runtime CPU) ─────────────────────────
print("⏳ Đang nạp mô hình VieNeu-TTS (sử dụng CPU / ONNX Runtime)...", flush=True)
try:
    try:
        # int8 giúp giảm 4x dung lượng RAM và tăng tốc độ xử lý trên CPU Cloud
        tts = Vieneu(precision="int8")
        print("💡 Đang chạy với chế độ tối ưu bộ nhớ precision=int8", flush=True)
    except Exception:
        tts = Vieneu()
    SR = getattr(tts, "sample_rate", 48000)
    print(f"✅ Mô hình đã sẵn sàng! Tần số mẫu: {SR} Hz", flush=True)
except Exception as e:
    print(f"❌ Lỗi khởi tạo mô hình: {e}", flush=True)
    tts = None
    SR = 48000

# ── Quản lý thư mục file giọng mẫu trên server ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Lấy danh sách toàn bộ 25 giọng chuẩn mặc định của VieNeu
def get_all_preset_voices():
    if tts is None:
        return [
            {"display": "Adam — Nam · Nam · Giọng đọc tự nhiên", "name": "Adam"},
            {"display": "Adam bựa — Nam · Bắc · Phong cách tự nhiên", "name": "Adam bựa"},
            {"display": "Hải Đăng — Nam · Bắc · Phong cách tự nhiên", "name": "Hải Đăng"},
        ]
    try:
        raw_voices = tts.list_preset_voices()
        all_v = []
        for label, vid in raw_voices:
            clean_label = label.replace("⭐ ", "").replace("⭐", "").strip()
            all_v.append({"display": clean_label, "name": vid})
        return all_v
    except Exception as e:
        print(f"Lỗi lấy danh sách giọng: {e}", flush=True)
        return [
            {"display": "Adam — Nam · Nam · Giọng đọc tự nhiên", "name": "Adam"},
            {"display": "Adam bựa — Nam · Bắc · Phong cách tự nhiên", "name": "Adam bựa"},
            {"display": "Hải Đăng — Nam · Bắc · Phong cách tự nhiên", "name": "Hải Đăng"},
        ]

ALL_PRESET_VOICES = get_all_preset_voices()
VOICE_MAP = {v["display"]: v["name"] for v in ALL_PRESET_VOICES}
ALL_DISPLAYS = [v["display"] for v in ALL_PRESET_VOICES]

# Đặt giọng Adam hoặc Hải Đăng làm mặc định ban đầu
DEFAULT_DISPLAY = next((d for d in ALL_DISPLAYS if "Adam —" in d), ALL_DISPLAYS[0] if ALL_DISPLAYS else "")

# Hàm lọc tìm kiếm giọng đọc theo từ khóa và phân loại
def filter_preset_voices(search_text, category):
    q = (search_text or "").strip().lower()
    filtered = []
    
    for v in ALL_PRESET_VOICES:
        disp = v["display"]
        disp_lower = disp.lower()
        
        # Lọc theo danh mục
        if category == "Giọng Nam" and "— Nam" not in disp and "· Nam" not in disp:
            continue
        elif category == "Giọng Nữ" and "— Nữ" not in disp and "· Nữ" not in disp:
            continue
        elif category == "Miền Bắc" and "Bắc" not in disp:
            continue
        elif category == "Miền Nam" and ("· Nam ·" not in disp and "— Nam · Nam" not in disp and "— Nữ · Nam" not in disp):
            continue
        elif category == "Miền Trung" and "Trung" not in disp:
            continue
            
        # Lọc theo từ khóa tìm kiếm (tên, vùng miền, phong cách)
        if q and q not in disp_lower:
            continue
            
        filtered.append(disp)
        
    if not filtered:
        return gr.Dropdown(choices=["(Không tìm thấy giọng phù hợp)"], value="(Không tìm thấy giọng phù hợp)")
    return gr.Dropdown(choices=filtered, value=filtered[0])

# Lấy danh sách các file âm thanh đã tải lên thư mục voices/
def get_custom_voice_files():
    valid_extensions = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
    if not os.path.exists(VOICES_DIR):
        return []
    files = [f for f in os.listdir(VOICES_DIR) if f.lower().endswith(valid_extensions)]
    files.sort()
    return files

SAMPLE_TEXT = (
    "Xin chào bạn! Đây là công cụ chuyển đổi văn bản thành giọng nói tiếng Việt tự động. "
    "Hệ thống hỗ trợ cả giọng đọc chuẩn của VieNeu lẫn tính năng sao chép chất giọng từ file âm thanh cá nhân."
)

# ── Hàm xử lý tạo giọng nói ────────────────────────────────────────────────
def process_tts(text, mode, preset_voice_display, custom_voice_file, temperature):
    if tts is None:
        raise gr.Error("❌ Mô hình chưa được tải thành công. Vui lòng kiểm tra lại console.")

    text = (text or "").strip()
    if not text:
        raise gr.Error("⚠️ Vui lòng nhập nội dung văn bản cần đọc!")

    start_time = time.perf_counter()

    if mode == "preset":
        # Kiểu 1: Giọng mặc định từ thư viện VieNeu
        if not preset_voice_display or "Không tìm thấy" in preset_voice_display:
            raise gr.Error("⚠️ Vui lòng chọn một giọng đọc hợp lệ trong danh sách!")
        
        # Tra cứu tên giọng thực tế (ví dụ: 'Adam bựa', 'Adam', 'Hải Đăng')
        actual_voice_name = VOICE_MAP.get(preset_voice_display, preset_voice_display.split(" — ")[0].strip())
        print(f"🎙️ Đang tổng hợp giọng mặc định: {actual_voice_name} ({preset_voice_display})", flush=True)
        
        audio = tts.infer(text, voice=actual_voice_name, temperature=float(temperature))
        info_source = f"Giọng chuẩn VieNeu: **{actual_voice_name}**"
    else:
        # Kiểu 2: Clone chất giọng từ file mẫu đã tải lên server
        if not custom_voice_file or "Chưa có file" in custom_voice_file:
            raise gr.Error("⚠️ Thư mục 'voices/' chưa có file .wav nào! Hãy copy file .wav (3-5s) vào thư mục voices/ trên máy.")

        ref_path = os.path.join(VOICES_DIR, custom_voice_file)
        if not os.path.exists(ref_path):
            raise gr.Error(f"⚠️ Không tìm thấy file âm thanh: {custom_voice_file}")

        print(f"🎙️ Đang clone chất giọng từ file: {ref_path}", flush=True)
        audio = tts.infer(text, ref_audio=ref_path, temperature=float(temperature))
        info_source = f"Clone từ file server: **{custom_voice_file}**"

    elapsed = time.perf_counter() - start_time
    audio_data = np.asarray(audio, dtype=np.float32)
    duration = len(audio_data) / SR if SR else 0.0

    stats = (
        f"✅ **Đã tạo giọng nói thành công!**\n\n"
        f"• **Nguồn giọng:** {info_source}\n"
        f"• **Thời gian xử lý:** **{elapsed:.2f}s**\n"
        f"• **Độ dài audio:** **{duration:.2f}s** (Tốc độ: {duration / elapsed:.1f}x thời gian thực)"
    )
    return (SR, audio_data), stats

# ── Hàm làm mới danh sách file mẫu ──────────────────────────────────────────
def refresh_file_list():
    files = get_custom_voice_files()
    if not files:
        return gr.Dropdown(choices=["Chưa có file mẫu trong thư mục voices/"], value="Chưa có file mẫu trong thư mục voices/")
    return gr.Dropdown(choices=files, value=files[0])

# Cập nhật hiển thị giao diện khi đổi chế độ
def on_mode_change(mode):
    if mode == "preset":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

# ── Thiết kế giao diện Web Responsive Mobile-First ─────────────────────────
custom_css = """
.container {
    max-width: 680px !important;
    margin: 0 auto !important;
    padding: 10px !important;
}
.btn-generate {
    height: 52px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
}
.audio-player {
    border-radius: 12px !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate"
)

with gr.Blocks(title="VietNeu Voice Studio") as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 12px;">
                <h1 style="font-size: 1.8rem; margin-bottom: 4px;">🎙️ VietNeu Voice Studio</h1>
                <p style="color: #64748b; font-size: 0.95rem;">Đọc văn bản & Clone giọng nói tiếng Việt bằng AI</p>
            </div>
            """
        )

        # 1. Ô nhập văn bản
        text_input = gr.Textbox(
            label="📝 1. Nhập văn bản cần đọc",
            placeholder="Nhập nội dung văn bản tiếng Việt tại đây...",
            value=SAMPLE_TEXT,
            lines=4,
            max_lines=8,
        )

        # 2. Chọn chế độ nguồn giọng
        mode_radio = gr.Radio(
            choices=[
                ("Kiểu 1: Giọng có sẵn (VieNeu)", "preset"),
                ("Kiểu 2: Giọng clone từ file (.wav)", "clone"),
            ],
            value="preset",
            label="🎯 2. Chọn kiểu giọng nói",
        )

        # Group Kiểu 1: Giọng dựng sẵn (Kèm thanh tìm kiếm & lọc nhanh)
        preset_box = gr.Group(visible=True)
        with preset_box:
            with gr.Row():
                search_box = gr.Textbox(
                    label="🔍 Tìm kiếm giọng đọc",
                    placeholder="Gõ để tìm: Adam, Hải Đăng, Quỳnh Anh, Bắc, Nam, Kể chuyện...",
                    lines=1,
                    max_lines=1,
                    scale=4,
                )
                clear_search_btn = gr.Button("✖️ Xoá tìm", scale=1, variant="secondary")

            category_pills = gr.Radio(
                choices=["Tất cả", "Giọng Nam", "Giọng Nữ", "Miền Bắc", "Miền Nam", "Miền Trung"],
                value="Tất cả",
                label="⚡ Lọc nhanh theo tiêu chí",
            )

            preset_dropdown = gr.Dropdown(
                label="🎙️ Danh sách giọng đọc (Có ô tìm kiếm trực tiếp bên trong)",
                choices=ALL_DISPLAYS,
                value=DEFAULT_DISPLAY,
                filterable=True,
                interactive=True,
            )

        # Group Kiểu 2: File mẫu trên server
        initial_files = get_custom_voice_files()
        initial_choices = initial_files if initial_files else ["Chưa có file mẫu trong thư mục voices/"]
        
        clone_box = gr.Group(visible=False)
        with clone_box:
            with gr.Row():
                custom_dropdown = gr.Dropdown(
                    label="Chọn file giọng mẫu trong thư mục voices/",
                    choices=initial_choices,
                    value=initial_choices[0],
                    allow_custom_value=True,
                    interactive=True,
                    scale=4,
                )
                refresh_btn = gr.Button("🔄 Quét lại", scale=1, variant="secondary")
            gr.Markdown(
                "💡 *Mẹo: File mẫu `.wav` chỉ cần dài **3–5 giây**, phát âm rõ ràng, ít ồn để clone chuẩn nhất.*"
            )

        # 3. Nút bấm tạo giọng
        generate_btn = gr.Button(
            "🔊 Bắt đầu tạo giọng nói",
            variant="primary",
            elem_classes=["btn-generate"],
        )

        # 4. Trình phát nhạc & thông tin kết quả
        audio_output = gr.Audio(
            label="🎧 Kết quả âm thanh (Nghe & Tải về)",
            type="numpy",
            autoplay=True,
            elem_classes=["audio-player"],
        )
        stats_output = gr.Markdown()

        # Tuỳ chọn nâng cao (thu gọn mặc định)
        with gr.Accordion("⚙️ Cài đặt nâng cao (Tuỳ chọn)", open=False):
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.8,
                step=0.05,
                label="Độ biểu cảm (Temperature) - Mặc định 0.8",
            )

    # ── Xử lý sự kiện tương tác ──────────────────────────────────────────────
    mode_radio.change(
        fn=on_mode_change,
        inputs=[mode_radio],
        outputs=[preset_box, clone_box],
    )

    # Tìm kiếm và lọc giọng đọc real-time
    search_box.input(
        fn=filter_preset_voices,
        inputs=[search_box, category_pills],
        outputs=[preset_dropdown],
    )
    category_pills.change(
        fn=filter_preset_voices,
        inputs=[search_box, category_pills],
        outputs=[preset_dropdown],
    )
    clear_search_btn.click(
        fn=lambda: ("", "Tất cả", gr.Dropdown(choices=ALL_DISPLAYS, value=DEFAULT_DISPLAY)),
        outputs=[search_box, category_pills, preset_dropdown],
    )

    refresh_btn.click(
        fn=refresh_file_list,
        outputs=[custom_dropdown],
    )

    generate_btn.click(
        fn=process_tts,
        inputs=[
            text_input,
            mode_radio,
            preset_dropdown,
            custom_dropdown,
            temperature_slider,
        ],
        outputs=[audio_output, stats_output],
    )

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    is_render = "RENDER" in os.environ or "PORT" in os.environ
    
    launch_res = demo.queue().launch(
        share=not is_render,
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
    )
    
    local_ip = get_local_ip()
    wifi_url = f"http://{local_ip}:{port}"
    share_url = getattr(demo, "share_url", None)
    if not share_url and isinstance(launch_res, tuple) and len(launch_res) > 2:
        share_url = launch_res[2]

    url_file = os.path.join(BASE_DIR, "urls.txt")
    with open(url_file, "w", encoding="utf-8") as f:
        f.write(f"LOCAL=http://127.0.0.1:{port}\nWIFI={wifi_url}\nSHARE={share_url}\n")
    print(f"\n=========================================\n🚀 LOCALHOST: http://127.0.0.1:{port}\n📱 WI-FI LAN: {wifi_url}\n🌐 PUBLIC SHARE: {share_url}\n=========================================\n", flush=True)
