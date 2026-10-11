import os
import sys
import socket

# Tối ưu hóa cực hạn bộ nhớ cho Cloud 512MB RAM
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["ORT_DISABLE_MEMORY_ARENA"] = "1"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import gc
import numpy as np
import soundfile as sf
import gradio as gr
from vieneu import Vieneu

# ── Khởi tạo mô hình VieNeu-TTS (Lazy Load + v3nano cho Cloud 512MB) ────────
_tts_instance = None
_tts_sr = 24000

def get_tts():
    global _tts_instance, _tts_sr
    if _tts_instance is None:
        print("⏳ Đang nạp mô hình VieNeu-TTS v3nano siêu nhẹ cho Cloud...", flush=True)
        gc.collect()
        try:
            # v3nano: 48M tham số, chỉ chiếm ~100MB RAM, chạy cực êm trên Render Free 512MB
            _tts_instance = Vieneu(mode="v3nano")
            _tts_sr = getattr(_tts_instance, "sample_rate", 24000)
            print(f"✅ Mô hình v3nano đã sẵn sàng! Tần số: {_tts_sr} Hz", flush=True)
        except Exception as e:
            print(f"⚠️ Thử khởi tạo với precision=int8: {e}", flush=True)
            try:
                _tts_instance = Vieneu(precision="int8")
                _tts_sr = getattr(_tts_instance, "sample_rate", 48000)
            except Exception as e2:
                print(f"⚠️ Thử khởi tạo mặc định: {e2}", flush=True)
                _tts_instance = Vieneu()
                _tts_sr = getattr(_tts_instance, "sample_rate", 48000)
    return _tts_instance, _tts_sr

# ── Quản lý thư mục file giọng mẫu và kết quả âm thanh ──────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Danh sách 11 giọng chuẩn chính xác của VieNeu v3nano
ALL_PRESET_VOICES = [
    {"display": "Adam — Nam · Bắc · Phong cách tự nhiên ⭐", "name": "Adam"},
    {"display": "Mai Anh — Nữ · Bắc · Phong cách nhẹ nhàng", "name": "Mai Anh"},
    {"display": "Trúc Ly — Nữ · Nam · Giọng đọc tự nhiên", "name": "Trúc Ly"},
    {"display": "Anh Khôi — Nam · Nam · Phong cách trẻ trung", "name": "Anh Khôi"},
    {"display": "Minh Quân — Nam · Nam · Giọng đọc truyền cảm", "name": "Minh Quân"},
    {"display": "Ái Hân — Nữ · Nam · Giọng đọc diễn cảm", "name": "Ái Hân"},
    {"display": "Mỹ Duyên — Nữ · Nam · Giọng đọc dịu dàng", "name": "Mỹ Duyên"},
    {"display": "Đức Trí — Nam · Bắc · Phong cách trầm ấm", "name": "Đức Trí"},
    {"display": "Hữu Quân — Nam · Bắc · Phong cách đĩnh đạc", "name": "Hữu Quân"},
    {"display": "Xuân Tiên — Nữ · Trung · Phong cách tự nhiên", "name": "Xuân Tiên"},
    {"display": "Mạnh Dũng — Nam · Bắc · Giọng đọc tin tức", "name": "Mạnh Dũng"},
]

VOICE_MAP = {v["display"]: v["name"] for v in ALL_PRESET_VOICES}
ALL_DISPLAYS = [v["display"] for v in ALL_PRESET_VOICES]
DEFAULT_DISPLAY = ALL_DISPLAYS[0]

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
def process_tts(text, mode, preset_voice_display, custom_voice_file, temperature, progress=gr.Progress(track_tqdm=True)):
    text = (text or "").strip()
    if not text:
        return None, "⚠️ **Vui lòng nhập nội dung văn bản cần đọc!**"

    progress(0.1, desc="⏳ Đang chuẩn bị mô hình AI...")
    try:
        engine, sr = get_tts()
    except Exception as e:
        return None, f"❌ **Không thể khởi động mô hình AI:** {e}"

    if engine is None:
        return None, "❌ **Lỗi:** Không thể nạp mô hình AI trên máy chủ!"

    start_time = time.perf_counter()

    if mode == "preset":
        if not preset_voice_display or "Không tìm thấy" in preset_voice_display:
            return None, "⚠️ **Vui lòng chọn một giọng đọc hợp lệ trong danh sách!**"
        
        actual_voice_name = VOICE_MAP.get(preset_voice_display, preset_voice_display.split(" — ")[0].strip())
        if actual_voice_name in ["Adam bựa", "Adam"]:
            actual_voice_name = "Adam"
            
        progress(0.4, desc=f"🎙️ Đang tổng hợp giọng {actual_voice_name}...")
        print(f"🎙️ Đang tổng hợp giọng: {actual_voice_name} ({preset_voice_display})", flush=True)
        try:
            audio = engine.infer(text, voice=actual_voice_name, temperature=float(temperature))
        except Exception as e:
            print(f"❌ Lỗi tổng hợp giọng: {e}", flush=True)
            return None, f"❌ **Lỗi khi tạo giọng:** {e}"
            
        info_source = f"Giọng chuẩn VieNeu: **{actual_voice_name}**"
    else:
        if not custom_voice_file or "Chưa có file" in custom_voice_file:
            return None, "⚠️ **Chưa có file âm thanh mẫu nào trong thư mục 'voices/'!**"

        ref_path = os.path.join(VOICES_DIR, custom_voice_file)
        if not os.path.exists(ref_path):
            return None, f"⚠️ **Không tìm thấy file âm thanh:** {custom_voice_file}"

        progress(0.4, desc="🎙️ Đang phân tích file mẫu và clone giọng...")
        print(f"🎙️ Đang clone chất giọng từ file: {ref_path}", flush=True)
        try:
            audio = engine.infer(text, ref_audio=ref_path, temperature=float(temperature))
        except Exception as e:
            print(f"❌ Lỗi clone giọng: {e}", flush=True)
            return None, f"❌ **Lỗi khi clone giọng từ file:** {e}"
            
        info_source = f"Clone từ file server: **{custom_voice_file}**"

    progress(0.85, desc="🎧 Đang xuất và lưu file âm thanh...")
    elapsed = time.perf_counter() - start_time
    audio_data = np.asarray(audio, dtype=np.float32)
    duration = len(audio_data) / sr if sr else 0.0

    # Lưu tệp .wav vào outputs/ để hỗ trợ iPhone mở trực tiếp hoặc tải về
    timestamp = int(time.time())
    out_filename = f"vietneu_{timestamp}.wav"
    out_filepath = os.path.join(OUTPUTS_DIR, out_filename)
    sf.write(out_filepath, audio_data, sr)

    gc.collect()

    progress(1.0, desc="✅ Hoàn tất!")
    stats = (
        f"✅ **Đã tạo giọng nói thành công!**\n\n"
        f"• **Nguồn giọng:** {info_source}\n"
        f"• **Thời gian xử lý:** **{elapsed:.2f}s**\n"
        f"• **Độ dài audio:** **{duration:.2f}s** (Tốc độ: {duration / elapsed:.1f}x thời gian thực)"
    )
    return out_filepath, stats

# ── Hàm làm mới danh sách file mẫu ──────────────────────────────────────────
def refresh_file_list():
    files = get_custom_voice_files()
    if not files:
        return gr.Dropdown(choices=["Chưa có file mẫu trong thư mục voices/"], value="Chưa có file mẫu trong thư mục voices/")
    return gr.Dropdown(choices=files, value=files[0])

# ── Hàm trích xuất văn bản từ nhiều định dạng file ──────────────────────────
def extract_text_from_file(file_obj):
    if file_obj is None:
        return gr.update(), ""
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    
    extracted_text = ""
    # 1. File Word (.docx)
    if ext == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            extracted_text = "\n".join(paragraphs)
        except Exception as e:
            return gr.update(), f"❌ Không thể đọc file Word (.docx): {e}"
            
    # 2. File PDF (.pdf)
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
            extracted_text = "\n".join(pages_text)
        except Exception as e:
            return gr.update(), f"❌ Không thể đọc file PDF: {e}"
            
    # 3. Các file text / config (.txt, .yml, .yaml, .md, .json, .csv, .log...)
    else:
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    extracted_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        if not extracted_text:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            except Exception as e:
                return gr.update(), f"❌ Không thể đọc file văn bản: {e}"

    clean_text = extracted_text.strip()
    if not clean_text:
        return gr.update(), f"⚠️ File `{filename}` không chứa nội dung chữ nào!"
        
    num_chars = len(clean_text)
    num_words = len(clean_text.split())
    status_msg = f"✅ **Đã nạp thành công từ `{filename}`** ({num_chars:,} ký tự, ~{num_words:,} từ)"
    return clean_text, status_msg

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
.action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 12px 14px;
    border-radius: 10px;
    font-size: 0.95rem;
    font-weight: 600;
    text-decoration: none !important;
    cursor: pointer;
    border: none;
    transition: all 0.2s ease;
    flex: 1 1 calc(50% - 8px);
    box-sizing: border-box;
    text-align: center;
}
.btn-ios { background: #0284c7; color: white !important; }
.btn-ios:hover { background: #0369a1; }
.btn-share { background: #10b981; color: white !important; }
.btn-share:hover { background: #059669; }
.btn-download { background: #475569; color: white !important; }
.btn-download:hover { background: #334155; }
"""

custom_head_html = """
<script>
let notificationsAllowed = false;

function playChime() {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        if (ctx.state === 'suspended') { ctx.resume(); }
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.setValueAtTime(587.33, ctx.currentTime);
        osc.frequency.setValueAtTime(880.0, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.25, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.6);
    } catch (e) {
        console.log("AudioContext error:", e);
    }
}

function requestNotificationAccess() {
    playChime();
    if ("Notification" in window) {
        Notification.requestPermission().then(permission => {
            const btn = document.getElementById("btn-notify-toggle");
            if (permission === "granted") {
                notificationsAllowed = true;
                if (btn) {
                    btn.innerHTML = "✅ Đã bật thông báo";
                    btn.style.background = "#10b981";
                }
                new Notification("🎙️ VietNeu Studio", {
                    body: "Đã bật chuông & thông báo thành công! Bạn sẽ nhận được thông báo khi tạo giọng xong.",
                    icon: "https://fav.farm/🔊"
                });
            } else {
                if (btn) btn.innerHTML = "⚠️ Chưa cấp quyền";
            }
        });
    } else {
        alert("Trình duyệt này hỗ trợ chuông báo âm thanh khi xong.");
    }
}

let currentAudioUrl = "";

function onAudioReady() {
    playChime();
    if (navigator.vibrate) {
        navigator.vibrate([200, 100, 250]);
    }
    const oldTitle = document.title;
    document.title = "🔔 [ĐÃ XONG!] " + oldTitle;
    setTimeout(() => { document.title = oldTitle; }, 10000);

    if ("Notification" in window && Notification.permission === "granted") {
        new Notification("🎉 VietNeu Voice Studio", {
            body: "Âm thanh của bạn đã được tạo xong! Bấm để mở nghe ngay.",
            icon: "https://fav.farm/🔊"
        });
    }

    setTimeout(() => {
        const audioTag = document.querySelector("#audio-result audio");
        if (audioTag && audioTag.src) {
            currentAudioUrl = audioTag.src;
            const openBtn = document.getElementById("btn-ios-open");
            const dlBtn = document.getElementById("btn-pc-download");
            const box = document.getElementById("ios-actions-box");
            if (openBtn) openBtn.href = currentAudioUrl;
            if (dlBtn) dlBtn.href = currentAudioUrl;
            if (box) box.style.display = "block";
        }
    }, 600);
}

async function shareAudioNative() {
    if (!currentAudioUrl) {
        const audioTag = document.querySelector("#audio-result audio");
        if (audioTag) currentAudioUrl = audioTag.src;
    }
    if (!currentAudioUrl) {
        alert("Chưa có file âm thanh để chia sẻ!");
        return;
    }
    try {
        const res = await fetch(currentAudioUrl);
        const blob = await res.blob();
        const file = new File([blob], "vietneu_voice.wav", { type: "audio/wav" });
        if (navigator.canShare && navigator.canShare({ files: [file] })) {
            await navigator.share({
                files: [file],
                title: "VietNeu TTS Audio",
                text: "Giọng nói tiếng Việt tạo bởi VietNeu Studio"
            });
        } else if (navigator.share) {
            await navigator.share({
                url: currentAudioUrl,
                title: "VietNeu TTS Audio"
            });
        } else {
            window.open(currentAudioUrl, "_blank");
        }
    } catch (e) {
        if (e.name !== "AbortError") {
            window.open(currentAudioUrl, "_blank");
        }
    }
}
</script>
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate"
)

with gr.Blocks(title="VietNeu Voice Studio", head=custom_head_html) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 12px;">
                <h1 style="font-size: 1.8rem; margin-bottom: 4px;">🎙️ VietNeu Voice Studio</h1>
                <p style="color: #64748b; font-size: 0.95rem;">Đọc văn bản & Clone giọng nói tiếng Việt bằng AI</p>
            </div>
            """
        )

        # Thanh kích hoạt thông báo khi xong
        gr.HTML(
            """
            <div id="notify-bar" style="display: flex; align-items: center; justify-content: space-between; background: #eff6ff; border: 1px solid #bfdbfe; padding: 10px 14px; border-radius: 12px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px; font-size: 0.88rem; color: #1e40af;">
                    <span style="font-size: 1.2rem;">🔔</span>
                    <span><b>Chuông & Thông báo:</b> Nhận báo hiệu khi tạo xong (kể cả khi đổi tab hoặc khoá màn hình)</span>
                </div>
                <button id="btn-notify-toggle" onclick="requestNotificationAccess()" style="background: #2563eb; color: white; border: none; padding: 6px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; cursor: pointer; white-space: nowrap;">
                    Bật thông báo
                </button>
            </div>
            """
        )

        # 1. Ô nhập văn bản & Nạp từ file
        text_input = gr.Textbox(
            label="📝 1. Nội dung văn bản cần đọc",
            placeholder="Nhập nội dung tiếng Việt hoặc nạp từ file (.txt, .docx, .yml, .pdf) ở bên dưới...",
            value=SAMPLE_TEXT,
            lines=4,
            max_lines=10,
        )

        with gr.Accordion("📂 Hoặc nạp từ file tài liệu (.txt, .docx, .yml, .yaml, .pdf, .md, .json...)", open=False):
            file_input = gr.File(
                label="Chọn hoặc kéo thả file văn bản (Tự động trích xuất nội dung vào ô trên)",
                file_types=[".txt", ".docx", ".yml", ".yaml", ".pdf", ".md", ".json", ".csv"],
                file_count="single",
            )
            file_status = gr.Markdown()

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
            label="🎧 Trình phát âm thanh (Nghe trực tiếp)",
            type="filepath",
            autoplay=True,
            elem_classes=["audio-player"],
            elem_id="audio-result",
        )
        stats_output = gr.Markdown()

        # Bảng điều khiển thao tác tệp thông minh & Tối ưu cho iPhone
        gr.HTML(
            """
            <div id="ios-actions-box" style="display: none; margin-top: 10px; padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;">
                <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 8px; color: #1e293b; display: flex; align-items: center; gap: 6px;">
                    <span>⚡ Thao tác tệp & Tải về:</span>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <a id="btn-ios-open" href="#" target="_blank" class="action-btn btn-ios">
                        📱 Mở tệp trực tiếp (iPhone)
                    </a>
                    <button id="btn-ios-share" class="action-btn btn-share" onclick="shareAudioNative()">
                        📤 Gửi Zalo / Lưu Tệp / AirDrop
                    </button>
                    <a id="btn-pc-download" href="#" download="vietneu_voice.wav" class="action-btn btn-download">
                        💾 Tải file .wav
                    </a>
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 8px; line-height: 1.4;">
                    💡 <b>Dành cho iPhone:</b> Bấm <i>"Mở tệp trực tiếp"</i> để Safari phát ngay trong trình phát iOS, hoặc bấm <i>"Gửi Zalo / Lưu Tệp"</i> để chuyển thẳng vào ứng dụng hoặc chọn thư mục lưu mà không phải vào mục Downloads!
                </div>
            </div>
            """
        )

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

    # Tự động trích xuất nội dung khi tải file văn bản lên
    file_input.change(
        fn=extract_text_from_file,
        inputs=[file_input],
        outputs=[text_input, file_status],
    )
    file_input.clear(
        fn=lambda: "",
        outputs=[file_status],
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
    ).then(
        None,
        None,
        None,
        js="() => { onAudioReady(); }"
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
    
    # Nạp sẵn mô hình trong luồng nền để khi người dùng bấm là có ngay
    import threading
    def prewarm_in_background():
        try:
            time.sleep(2)
            print("🔥 Bắt đầu nạp sẵn mô hình AI trong luồng nền...", flush=True)
            get_tts()
            print("🔥 Nạp sẵn mô hình hoàn tất! Sẵn sàng phục vụ tức thì.", flush=True)
        except Exception as e:
            print(f"Lỗi nạp nền: {e}", flush=True)
    threading.Thread(target=prewarm_in_background, daemon=True).start()

    launch_res = demo.queue().launch(
        share=not is_render,
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
        allowed_paths=[VOICES_DIR, OUTPUTS_DIR],
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
