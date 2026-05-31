
### Các bước chạy hệ thống:
1. **Mở Terminal (Command Prompt / PowerShell)** và trỏ vào thư mục `ted_ai_app/backend`.
2. **Cài đặt thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Khởi động Server**:
   Bạn có thể chạy trực tiếp bằng lệnh:
   ```bash
   python main.py
   ```
   *(Hoặc click đúp vào file `run.bat` ngoài thư mục gốc để chạy tự động).*
4. **Truy cập ứng dụng**:
   Mở trình duyệt và truy cập vào địa chỉ: [http://localhost:8000](http://localhost:8000).

---

## 2. Luồng xử lý dữ liệu

Khi người dùng tải lên một (hoặc nhiều) video, hệ thống sẽ thực hiện một luồng pipeline tự động hoàn toàn như sau:

1. **Upload Video & Tiền xử lý (Upload & Preprocessing)**
   - Người dùng tải lên một video định dạng `.mp4` từ Frontend. Backend (sử dụng **FastAPI**) nhận file qua API endpoint `/upload`. File video này được lưu tạm vào thư mục `uploads/`.

2. **Nhận diện Giọng nói (Speech-to-Text)**
   - **Mô hình sử dụng**: **OpenAI Whisper**.
   - **Quy trình chi tiết**: Hệ thống sử dụng `imageio_ffmpeg` (ffmpeg) để trích xuất âm thanh từ file `.mp4`. Sau đó, Whisper model xử lý âm thanh này và chuyển đổi thành văn bản. Trả về `full_text` (toàn bộ nội dung bài nói) và `segments` (các đoạn nhỏ kèm mốc thời gian làm phụ đề).

3. **Đánh giá Độ khó Ngôn ngữ (CEFR Classification)**
   - **Mô hình sử dụng**: **TF-IDF Vectorizer + LinearSVC / Hybrid XGBoost** (lưu ở `hybrid_bundle_v2.pkl`).
   - **Quy trình chi tiết**: `full_text` được đưa qua TF-IDF để phân tích độ phức tạp của từ vựng (Lexical Richness). Mô hình phân loại sẽ vạch ra ranh giới quyết định để xếp văn bản vào một trong 6 cấp độ chuẩn Châu Âu: **A1, A2, B1, B2, C1, C2**. *(Hệ thống có cơ chế fallback đếm tổng số từ và tỉ lệ từ vựng độc nhất nếu chưa load được model).*

4. **Phân loại Chủ đề (Topic Classification)**
   - **Mô hình sử dụng**: **TF-IDF Vectorizer + LinearSVC** (lưu ở `yahoo_10topics_tfidf_linearsvc.joblib`).
   - **Quy trình chi tiết**: TF-IDF Vectorizer biến văn bản thành ma trận vector đa chiều. Thuật toán **Support Vector Machine (nhân tuyến tính)** xử lý ma trận này để xác định chủ đề dựa vào từ khóa đặc trưng nhất. Mô hình trả về 1 trong 10 chủ đề chính, sau đó hệ thống sẽ map lại ra các category chuẩn (*Technology, Education, Business, Health, Science...*).

5. **Tạo Bài tập Tự động (Question Generation)**
   - **Mô hình sử dụng**: **T5 (Text-to-Text Transfer Transformer)** (`valhalla/t5-base-qg-hl` - fine-tune trên SQuAD).
   - **Quy trình chi tiết**:
     - *Trích xuất ứng viên đáp án*: Dùng Regex/NLP tìm các "ứng viên" đáp án (Số liệu, thuật ngữ khoa học, cụm danh từ).
     - *Sinh MCQ*: Chọn 1 đáp án, đánh dấu token `<hl>`, đưa vào T5 prompt `generate question:` để tự sinh câu hỏi. Tìm distractors (đáp án sai) từ các ứng viên khác có cùng thể loại.
     - *Sinh Cloze Test (Điền khuyết)*: Tìm đoạn văn dài/khó nhất, khoét 10 chỗ trống tương ứng 10 từ khóa, trộn các lựa chọn để tạo bài đọc hiểu đục lỗ chuẩn thi THPT.

6. **Trích xuất Từ vựng Chuyên ngành (Vocabulary Extraction)**
   - **Mô hình/Công cụ**: **NLTK (POS Tagging), TextBlob**, **GoogleTranslator** và **eng_to_ipa**.
   - **Quy trình chi tiết**: Dùng TextBlob/NLTK phân tích POS Tagging để lọc **Danh từ (NN)** và **Tính từ (JJ)**. Tính điểm của từ dựa trên (Tần suất xuất hiện) x (Trọng số độ khó CEFR của từ). Lọc ra Top 15 từ vựng có điểm cao nhất, lấy phiên âm chuẩn và dịch nghĩa sang tiếng Việt.

7. **Lưu trữ & Phản hồi**
   - **Quy trình chi tiết**: Toàn bộ dữ liệu tổng hợp (Transcript, CEFR, Topic, Bài tập MCQ & Cloze test, Vocab, Video URL) được lưu vào cơ sở dữ liệu nội bộ (`db.json`) và trả về JSON. Frontend nhận cục dữ liệu này và render lên giao diện SPA.

---

## 3. Chi tiết về các Mô hình Học Máy tự huấn luyện

Dự án này sử dụng các mô hình Học Máy truyền thống (Traditional Machine Learning) kết hợp với Transformer để xử lý ngôn ngữ tự nhiên (NLP) tối ưu hiệu năng mà không cần đến tài nguyên GPU khổng lồ liên tục.

### A. Mô hình `TF-IDF Vectorizer + LinearSVC tag Topic`
**(Đóng gói trong file: `yahoo_10topics_tfidf_linearsvc.joblib`)**

- **Mục tiêu**: Phân loại văn bản transcript vào một trong 10 chủ đề chính (ví dụ: *Business & Finance, Computers & Internet, Health, Science & Mathematics, Politics & Government...*).
- **Bộ máy Vector hoá (TF-IDF Vectorizer)**: Thuật toán *Term Frequency-Inverse Document Frequency* đếm số lần xuất hiện của các từ khóa (TF), phạt (chia cho IDF) đối với những từ phổ biến (the, a, an). Kết quả là ma trận thưa (sparse matrix) chứa đặc trưng từ khóa biểu tượng của chủ đề.
- **Bộ phân loại (LinearSVC)**: Thuật toán *Support Vector Machine (nhân tuyến tính)* nhận ma trận thưa. Điểm mạnh của LinearSVC là xử lý không gian vector từ vựng chục ngàn chiều, tìm ra "siêu phẳng" (hyperplanes) có biên độ an toàn lớn nhất để cắt và phân tách 10 nhóm chủ đề cực kỳ chính xác.
- **Output**: Trả về nhãn chủ đề khớp nhất.

### B. Mô hình `TF-IDF Vectorizer + LinearSVC / XGBoost tag CEFR`
**(Đóng gói trong file: `hybrid_bundle_v2.pkl`)**

- **Mục tiêu**: Đánh giá độ khó của văn bản theo chuẩn quy chiếu ngôn ngữ Châu Âu (CEFR: A1 → C2).
- **Bộ máy Vector hoá (TF-IDF)**: Nhạy bén với **Độ phức tạp của từ vựng (Lexical Richness)**. Các từ vựng bậc cao (VD: *Hyponatremia, Intoxication*) được gán trọng số cực cao, từ cấp A1/A2 bị xem nhẹ.
- **Bộ phân loại (LinearSVC / Hybrid XGBoost)**: Dựa vào sự xuất hiện của các cụm từ khó (n-grams) và cấu trúc câu phức tạp, thuật toán vạch ra ranh giới quyết định. VD: Transcript có mật độ từ vựng dày đặc và ít gặp bị đẩy sang vùng `C1` hoặc `C2`. Đoạn văn ngắn, từ lặp lại sẽ rơi vào `A2`.
- **Output**: Chuỗi ký tự chuẩn CEFR để đề xuất lộ trình học phù hợp.
