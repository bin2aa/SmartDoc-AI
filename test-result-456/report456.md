# Report (Task 8.2.4 - 8.2.6)

**Ngày báo cáo:** 2026-04-15

---

## 1. Cải thiện Chiến lược Chia nhỏ Văn bản (8.2.4)
Hệ thống đã được cấu hình để cho phép linh hoạt trong việc xử lý văn bản đầu vào.

- **Tính năng đã thực hiện:**
    - Cập nhật giao diện **Settings** cho phép người dùng tùy chỉnh `chunk_size` (500 - 2000) và `chunk_overlap` (50 - 300).
    - Duy trì công cụ **Chunk Strategy Benchmark** để đo lường độ chính xác (Accuracy Proxy) dựa trên từ khóa cho từng bộ tham số.
- **Kết quả:** Người dùng có thể tối ưu hóa hiệu suất truy xuất dựa trên đặc thù của từng loại tài liệu (ví dụ: tài liệu pháp lý cần chunk lớn hơn tài liệu kỹ thuật).

---

## 2. Citation và Source Tracking Nâng cao (8.2.5)
Nâng cấp khả năng minh bạch thông tin, giúp người dùng kiểm chứng câu trả lời của AI.

- **Tính năng đã thực hiện:**
    - **Lưu trữ Metadata chi tiết:** Ngoài tên file, hệ thống hiện lưu trữ số trang (page number), vị trí chunk và nội dung văn bản gốc vào lịch sử chat.
    - **UI Source Details:** Thay thế danh sách nguồn đơn giản bằng một Expander "Xem chi tiết nguồn tham khảo" phong phú.
    - **Highlight Ngữ cảnh:** Tự động đánh dấu (highlight) các đoạn văn bản được AI sử dụng trực tiếp để tạo câu trả lời.
    - **Truy cập trực tiếp:** Cung cấp link mở file nguồn ngay trong phần chi tiết.

---

## 3. Conversational RAG & Query Rewriting (8.2.6)
Tối ưu hóa khả năng hội thoại liên tục, giúp AI hiểu các câu hỏi tiếp nối (follow-up questions).

- **Tính năng đã thực hiện:**
    - **LLM-based Query Rewriting:** Sử dụng LLM để phân tích lịch sử hội thoại và viết lại các câu hỏi ngắn hoặc mơ hồ (như "Còn nó thì sao?", "Giải thích thêm về phần này") thành các câu hỏi độc lập (Standalone Questions) đầy đủ ngữ cảnh.
    - **Hiển thị Rewritten Query:** Hiển thị công khai câu hỏi đã được tối ưu trong phần "Xem chi tiết nguồn tham khảo" để người dùng biết AI đã hiểu ý mình như thế nào.
    - **Hybrid Retrieval cho Follow-up:** Câu hỏi sau khi viết lại sẽ được đưa qua pipeline tìm kiếm (Vector + BM25 + Rerank) để đảm bảo độ chính xác cao nhất.

---
