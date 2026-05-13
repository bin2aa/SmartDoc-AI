# Tóm tắt cải tiến OCR

Ngày: 2026-05-13

## Tổng quan

Đã triển khai 3 cải tiến OCR nhằm giảm tình trạng OCR đọc được chữ nhưng trả lời sai:

1. Tự động nhận diện PDF scan và bỏ qua OCR khi PDF có text thật.
2. Cấu hình OCR trong UI (ngôn ngữ, DPI, PSM, OEM, tiền xử lý).
3. OCR theo từng trang, có metadata trang để cải thiện trích dẫn và truy hồi.

## Thay đổi chính

### 1) Chỉ OCR cho PDF scan

- Khi bật OCR cho PDF, hệ thống thử trích xuất text thường trước.
- Nếu PDF đã có text đủ tốt (độ dài và tỉ lệ ký tự chữ đạt ngưỡng), OCR sẽ bị bỏ qua.
- Nếu text kém hoặc lỗi trích xuất, OCR sẽ tự động chạy.

Logic chính:

- Kiểm tra chất lượng text dựa trên số ký tự và tỉ lệ ký tự chữ.
- Ngưỡng được định nghĩa trong constants để đảm bảo nhất quán.

### 2) Cấu hình OCR trong Settings

Thêm panel cấu hình OCR, cho phép tinh chỉnh:

- Language pack (ví dụ: "vie+eng")
- DPI khi render PDF
- Page segmentation mode (PSM)
- OCR engine mode (OEM)
- Bật/tắt tiền xử lý ảnh (grayscale + contrast + denoise + binarize)
- Tự động chỉ OCR cho PDF scan

Tất cả cấu hình được lưu xuống đĩa và khôi phục khi khởi động.

### 3) OCR theo từng trang

- OCR trả về text theo từng trang thay vì gộp toàn bộ.
- Mỗi trang OCR trở thành một document có số trang.
- Chunk metadata có thông tin trang để trích dẫn và truy hồi chính xác hơn.

## Cách sử dụng

1. Vào Settings và điều chỉnh OCR Configuration nếu cần.
2. Ở Documents, bật OCR khi upload PDF scan hoặc ảnh.
3. Nếu Auto OCR bật, PDF có text thật sẽ tự bỏ qua OCR.

## File đã cập nhật

- src/utils/ocr_utils.py
  - Thêm tiền xử lý và OCR theo trang.
  - Thêm kiểm tra chất lượng text không OCR.

- src/services/document_service.py
  - Tự động nhận diện PDF scan.
  - OCR theo trang và metadata trang.
  - Áp dụng cấu hình OCR vào quá trình trích xuất.

- src/views/ocr_settings.py
  - UI cấu hình OCR mới.

- src/views/settings_screen.py
  - Thêm mục OCR settings.
  - Lưu cấu hình OCR.

- app.py
  - Khởi tạo OCR settings trong session state.

- src/controllers/document_controller.py
  - Truyền OCR config khi xử lý.
  - Lưu OCR config vào metadata tài liệu đã load.

- src/utils/constants.py
  - Thêm default OCR và ngưỡng chất lượng.

## Ghi chú

- Độ chính xác OCR vẫn phụ thuộc chất lượng tài liệu.
- Nếu trả lời vẫn sai, hãy kiểm tra nguồn truy hồi và cân nhắc bật hybrid search hoặc reranking.
