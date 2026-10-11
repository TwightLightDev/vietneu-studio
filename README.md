# 🎙️ VietNeu Voice Studio

Công cụ Text-to-Speech (TTS) và Clone giọng nói tiếng Việt tối ưu cho Web và Cloud, sử dụng thư viện VieNeu-TTS (ONNX Runtime).

## Tính năng chính
- **Đọc văn bản (Preset voices):** 25 giọng tiếng Việt đa dạng (Bắc, Trung, Nam, Adam, Adam bựa, Hải Đăng,...).
- **Tìm kiếm giọng nói & bộ lọc danh mục:** Lọc theo giới tính và vùng miền trực tiếp trên giao diện.
- **Clone giọng mẫu:** Tải các file `.wav` vào thư mục `voices/` để tạo giọng đọc tương tự.
- **Tối ưu bộ nhớ:** Tự động hỗ trợ `precision="int8"` nhằm tiết kiệm RAM trên Cloud Free Tier (Render.com).

## Triển khai trên Render.com
1. Tạo Web Service mới trên [Render Dashboard](https://dashboard.render.com).
2. Kết nối repo này hoặc nhập URL repository.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python app.py`
5. Chọn gói **Free**.
