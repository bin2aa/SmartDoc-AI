# Update 9-10 (2026-04-14)

## Mục tiêu đợt cập nhật

Đợt này hoàn thiện cả Task 9 và Task 10 theo hướng chạy thực tế end-to-end:

1. Task 9: Re-ranking với Cross-Encoder, A/B metrics và benchmark nhiều query.
2. Task 10: Kích hoạt Advanced RAG với Self-RAG trong luồng Chat UI và API.

---

## Tổng quan kết quả

- Task 9: Đã hoàn thiện trước đó và vẫn giữ nguyên.
- Task 10: Đã nối vào flow chính, không còn ở trạng thái "hàm có sẵn nhưng chưa được gọi".

---

## Task 9 (Re-ranking) - Tóm tắt đã có

### 1) A/B comparison tự động trong retrieval

**File:** `src/controllers/chat_controller.py`

- Có so sánh `hybrid_vs_vector`.
- Có so sánh `rerank_vs_biencoder`.
- Lưu metrics vào `st.session_state.retrieval_comparison`.

### 2) Đồng bộ non-stream + stream

**File:** `src/controllers/chat_controller.py`

- Cả `process_query(...)` và `process_query_stream(...)` đều dùng cùng logic comparison.

### 3) Benchmark nhiều query

**File:** `src/controllers/chat_controller.py`

- API benchmark: `benchmark_rerank_queries(queries, k)`.
- Trả chi tiết từng query + summary trung bình (`avg_bi_encoder_ms`, `avg_rerank_ms`, `avg_rerank_only_ms`, `avg_rank_changes`).

### 4) UI metrics + benchmark

**File:** `src/views/chat_screen.py`

- Tách hiển thị `Hybrid vs Vector` và `Cross-Encoder Re-rank vs Bi-Encoder`.
- Thêm panel `Re-ranking Benchmark` để chạy benchmark trực tiếp trên UI.

### 5) Giải thích các thông số so sánh (cách đọc kết quả)

Phần này giúp diễn giải đúng các số liệu trong `Retrieval Metrics` và `Re-ranking Benchmark`.

#### A. Nhóm Hybrid vs Vector

- `vector_results`: số chunk trả về từ truy xuất vector thuần (không BM25).
- `hybrid_results`: số chunk trả về khi bật hybrid (vector + BM25 fusion).
- `vector_ms`: tổng thời gian truy xuất ở chế độ vector thuần.
- `hybrid_ms`: tổng thời gian truy xuất ở chế độ hybrid.
- `overlap`: số chunk giao nhau giữa 2 danh sách kết quả.

Diễn giải nhanh:

- `overlap` thấp => hybrid đang kéo thêm ngữ cảnh khác so với vector thuần.
- `hybrid_ms` cao hơn `vector_ms` là bình thường vì có thêm bước BM25/fusion.

#### B. Nhóm Cross-Encoder Re-rank vs Bi-Encoder

- `mode`: cấu hình nền được dùng khi so sánh (`vector+rerank` hoặc `hybrid+rerank`).
- `bi_encoder_results`: số chunk top-k trước khi re-rank.
- `rerank_results`: số chunk top-k sau khi re-rank (thường bằng k).
- `bi_encoder_ms`: thời gian pipeline không bật re-rank.
- `rerank_ms`: tổng thời gian pipeline có re-rank.
- `rerank_only_ms`: phần thời gian tăng thêm do riêng bước cross-encoder re-rank.
- `topk_overlap`: số phần tử chung giữa top-k trước/sau re-rank.
- `topk_rank_changes`: số vị trí trong top-k bị đổi thứ tự sau re-rank.

Diễn giải nhanh:

- `topk_rank_changes` càng cao => re-rank can thiệp càng mạnh vào thứ hạng.
- `rerank_only_ms` là chi phí latency "thật" của cross-encoder, dùng để cân bằng giữa chất lượng và tốc độ.
- Nếu `topk_rank_changes` gần 0 trong nhiều query, có thể giảm tần suất re-rank để tối ưu hiệu năng.

#### C. Nhóm Benchmark summary (nhiều query)

- `queries`: số lượng query đã chạy benchmark.
- `avg_bi_encoder_ms`: thời gian trung bình không bật re-rank.
- `avg_rerank_ms`: thời gian trung bình có re-rank.
- `avg_rerank_only_ms`: thời gian trung bình chỉ của bước re-rank.
- `avg_rank_changes`: số thay đổi thứ hạng trung bình trên top-k.
- `mode`: benchmark đang chạy theo nền `vector` hay `hybrid`.

Gợi ý đọc kết quả khi ra quyết định:

1. Nếu `avg_rank_changes` tăng rõ và câu trả lời thực tế tốt hơn, chấp nhận `avg_rerank_only_ms` cao hơn.
2. Nếu latency tăng nhiều nhưng `avg_rank_changes` thấp, re-rank có thể chưa đáng bật mặc định.
3. Nên so sánh trên cùng bộ query đại diện nghiệp vụ, không kết luận bằng 1-2 câu hỏi lẻ.
4. Nên chạy 2-3 lần và bỏ lần đầu (warm-up) để giảm sai lệch do tải model/cache.

---

## Task 10 (Advanced RAG với Self-RAG) - Các thay đổi mới

### 1) Bật/tắt Self-RAG từ Settings và lưu persistent

**Files:**

- `src/views/settings_screen.py`
- `app.py`

Chi tiết:

- Thêm toggle mới: `Self-RAG (Eval + Confidence)` trong Retrieval Strategy.
- Lưu setting `use_self_rag` vào file settings JSON.
- Khởi tạo session state `use_self_rag` từ persisted settings khi app start.

Ý nghĩa:

- Người dùng bật/tắt Self-RAG mà không cần sửa code.
- Trạng thái toggle được giữ sau khi reload app.

### 2) Nối Self-RAG vào luồng Chat UI thực tế

**File:** `src/views/chat_screen.py`

Chi tiết:

- Trong `_render_chat_input()`:
  - Nếu `use_self_rag=True`: gọi `process_query_with_self_rag(...)`.
  - Nếu `use_self_rag=False`: giữ luồng stream cũ `process_query_stream(...)`.
- Khi chạy Self-RAG, UI hiển thị:
  - confidence score (%),
  - confidence level,
  - self-evaluation justification.
- Metadata Self-RAG (`used_self_rag`, `confidence_score`, `confidence_level`, `self_eval_justification`) được lưu vào chat history.
- Thêm hàm `_render_self_rag_metadata(...)` để hiển thị lại các thông tin này khi render lịch sử chat.

Ý nghĩa:

- 4 yêu cầu Task 10 không chỉ tồn tại ở controller mà đã xuất hiện trong UX thực tế của app.

### 3) Nối Self-RAG vào API và bỏ confidence hardcode

**Files:**

- `src/api/models.py`
- `src/api/server.py`

Chi tiết model API:

- `QueryRequest`: thêm `use_self_rag: bool = False`.
- `BatchQueryRequest`: thêm `use_self_rag: bool = False` cho toàn bộ batch.
- `QueryResponse`: thêm trường optional:
  - `confidence_level`
  - `self_evaluation`

Chi tiết endpoint `/api/query`:

- Nếu `use_self_rag=True`:
  - gọi `chat_controller.process_query_with_self_rag(...)`.
  - nhận confidence dạng 0-100 và chuẩn hóa về 0-1 để phù hợp schema API.
  - trả kèm `confidence_level` và `self_evaluation`.
- Nếu `use_self_rag=False`:
  - giữ flow cũ `process_query(...)`.

Chi tiết endpoint `/api/batch-query`:

- Truyền `use_self_rag` xuống từng query request trong batch.

Ý nghĩa:

- API client có thể bật/tắt Self-RAG theo request.
- Confidence trong API không còn luôn cố định 0.85 khi dùng Self-RAG.

---

## Đánh giá mức đáp ứng yêu cầu Task 10 sau cập nhật

1. Implement Self-RAG: LLM tự đánh giá câu trả lời

- Đã active trong flow thực tế khi bật `use_self_rag`.

2. Query rewriting: Tự động cải thiện câu hỏi

- Đã active (vốn có sẵn), nay nằm trong luồng Self-RAG đang được gọi thật.

3. Multi-hop reasoning

- Đã active theo điều kiện query phức tạp trong `process_query_with_self_rag(...)`.

4. Confidence scoring

- Đã active và được hiển thị ở UI, trả về qua API.

Kết luận: Task 10 đã chuyển từ mức "implemented in code only" sang "implemented + wired + observable".

---

## Cách kiểm thử nhanh sau cập nhật

### A. Kiểm thử Task 9 (Re-ranking)

1. Vào Settings, bật `Cross-Encoder Re-ranking` (và tùy chọn bật thêm Hybrid).
2. Hỏi một câu trong tab Chat, mở `Retrieval Metrics`:
   - xác nhận có mục `Cross-Encoder Re-rank vs Bi-Encoder`.
3. Mở `Re-ranking Benchmark`:
   - nhập 3-5 query đại diện,
   - chạy benchmark,
   - kiểm tra summary + bảng kết quả.

### B. Kiểm thử Task 10 (Self-RAG)

1. Vào Settings, bật `Self-RAG (Eval + Confidence)`.
2. Hỏi một câu phức tạp (có yếu tố so sánh/tổng hợp nhiều ý).
3. Xác nhận UI hiển thị:
   - `Self-RAG confidence: ...% (... level ...)`
   - `Self-evaluation: ...`
4. Refresh app và kiểm tra toggle Self-RAG vẫn giữ trạng thái đã lưu.

### C. Kiểm thử API Self-RAG

Ví dụ request:

```json
{
  "query": "So sánh điểm khác nhau giữa phương án A và B trong tài liệu",
  "k": 3,
  "temperature": 0.7,
  "use_self_rag": true
}
```

Kỳ vọng response có thêm:

- `confidence` (động, chuẩn hóa 0..1)
- `confidence_level`
- `self_evaluation`

---

## Danh sách file đã chỉnh sửa trong đợt này

- `app.py`
- `src/views/settings_screen.py`
- `src/views/chat_screen.py`
- `src/api/models.py`
- `src/api/server.py`
- `update_9_10.md`
