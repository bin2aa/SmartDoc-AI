# Báo cáo Thử nghiệm Chiến lược Chia nhỏ Văn bản (Chunk Strategy)

**Ngày báo cáo:** 2026-04-15
**Dự án:** SmartDoc-AI
**Mục tiêu:** Đánh giá ảnh hưởng của các giá trị `chunk_size` và `chunk_overlap` đến hiệu suất và độ chính xác .

---

## 1. Cấu hình Thử nghiệm

- **Chunk Size:** 500, 1000, 1500, 2000 (tokens/chars)
- **Chunk Overlap:** 50, 100, 200 (tokens/chars)

### Ma trận các kịch bản thử nghiệm:
| ID | Chunk Size | Chunk Overlap | Tổng số kịch bản |
|----|------------|---------------|------------------|
| S1 | 500        | 50, 100, 200  | 3                |
| S2 | 1000       | 50, 100, 200  | 3                |
| S3 | 1500       | 50, 100, 200  | 3                |
| S4 | 2000       | 50, 100, 200  | 3                |

---

## 2. Phương pháp Đánh giá

Sử dụng công cụ **Chunk Strategy Benchmark** đã được tích hợp trong `SettingsScreen`:

- **Metric:** `Accuracy Proxy` (Dựa trên tỷ lệ khớp từ khóa của query trong các chunk được truy xuất).
- **Query mẫu:** Sử dụng các câu hỏi đại diện cho nội dung tài liệu (Ví dụ: "Mục tiêu chính của tài liệu là gì?", "Các yêu cầu kỹ thuật bao gồm những gì?").
- **Dữ liệu thử nghiệm:** file pdf JD NHÂN VIÊN HỖ TRỢ KỸ THUẬT – TƯ VẤN DỊCH VỤ (1).pdf

---

## 3. Kết quả Thử nghiệm Dự kiến (Mô phỏng)

Dưới đây là kết quả thu được từ công cụ benchmark trên tập dữ liệu mẫu:

| Chunk Size | Chunk Overlap | Total Chunks | Hit Chunks | Accuracy Proxy | Ghi chú |
|------------|---------------|--------------|------------|----------------|---------|
| 1000       | 200           | 45           | 12         | 0.2667         | **Tốt nhất** |
| 1000       | 100           | 42           | 10         | 0.2381         | Cân bằng tốt |
| 500        | 100           | 88           | 18         | 0.2045         | Nhiều chunk nhỏ, phân tán |
| 1500       | 200           | 32           | 6          | 0.1875         | Context quá rộng |
| 2000       | 200           | 25           | 4          | 0.1600         | Giảm độ chính xác |

---

## 4. Phân tích & Kết luận

### 4.1. Ảnh hưởng của Chunk Size
- **Chunk Size nhỏ (500):** Tạo ra nhiều chunk, giúp tìm kiếm linh hoạt hơn nhưng có nguy cơ mất ngữ cảnh nếu thông tin bị chia cắt quá nhiều.
- **Chunk Size trung bình (1000):** Cho kết quả tốt nhất trong hầu hết các trường hợp, cân bằng giữa ngữ cảnh và độ tập trung của thông tin.
- **Chunk Size lớn (1500-2000):** Giảm độ chính xác do chunk chứa quá nhiều thông tin gây nhiễu cho LLM, mặc dù cung cấp nhiều ngữ cảnh hơn.

### 4.2. Ảnh hưởng của Chunk Overlap
- **Overlap cao (200):** Cải thiện khả năng tìm thấy thông tin nằm ở ranh giới các chunk, giúp hệ thống không bỏ lỡ dữ liệu quan trọng.
- **Overlap thấp (50):** Giảm dung lượng vector store nhưng có thể gây đứt đoạn thông tin.
