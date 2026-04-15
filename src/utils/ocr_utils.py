import os
import platform
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# 1. Cấu hình tự động theo Hệ điều hành
OS_NAME = platform.system()

# Chúng ta ưu tiên lấy từ file .env, nếu không có mới dùng mặc định
TESSERACT_CMD = os.getenv('TESSERACT_CMD', None)
POPPLER_PATH = os.getenv('POPPLER_PATH', None)

if OS_NAME == "Windows":
    # Nếu là Windows mà chưa cấu hình trong .env, dùng đường dẫn mặc định
    if not TESSERACT_CMD:
        TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not POPPLER_PATH:
        # ĐƯỜNG DẪN ĐÚNG BẠN ĐÃ SỬA
        POPPLER_PATH = r'C:\poppler-25.12.0\Library\bin'
        
# Gán đường dẫn cho Tesseract (Nếu có)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_with_ocr(file_path: str, lang: str = 'vie+eng') -> str:
    """
    Đọc file PDF dạng ảnh HOẶC file ảnh trực tiếp và trích xuất văn bản bằng OCR (Đa nền tảng).
    CÓ TÍCH HỢP DEBUG LOG ĐỂ XEM TRỰC TIẾP KẾT QUẢ.
    """
    extracted_text = ""
    extension = os.path.splitext(file_path)[1].lower()
    
    try:
        print(f"\n[{OS_NAME}] Đang xử lý OCR cho file: {file_path}...")
        
        # --- TRƯỜNG HỢP 1: FILE LÀ PDF ---
        if extension == '.pdf':
            # 2. Xử lý đường dẫn Poppler
            pdf_kwargs = {'dpi': 300}
            if POPPLER_PATH and OS_NAME == "Windows":
                # Check thêm một lớp an toàn, báo lỗi tiếng Việt nếu sai đường dẫn
                if not os.path.exists(POPPLER_PATH):
                    raise FileNotFoundError(f"Không tìm thấy thư mục Poppler tại: {POPPLER_PATH}")
                pdf_kwargs['poppler_path'] = POPPLER_PATH
                
            pages = convert_from_path(file_path, **pdf_kwargs)
            
            for i, page_image in enumerate(pages):
                print(f"[{OS_NAME}] Đang OCR trang {i + 1}/{len(pages)}...")
                text = pytesseract.image_to_string(page_image, lang=lang, config='--psm 6')
                
                # ================= DEBUG LOG DÀNH CHO PDF =================
                print(f"\n[🔍 DEBUG OCR] KẾT QUẢ TRANG {i + 1}:")
                print("-" * 50)
                print(text.strip() if text.strip() else "[KHÔNG ĐỌC ĐƯỢC CHỮ NÀO]")
                print("-" * 50 + "\n")
                # ==========================================================

                extracted_text += f"\n\n--- Bắt đầu trang {i + 1} ---\n\n"
                extracted_text += text
                
        # --- TRƯỜNG HỢP 2: FILE LÀ ẢNH (PNG, JPG,...) ---
        else:
            print(f"[{OS_NAME}] Đang OCR từ file ảnh trực tiếp...")
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang=lang, config='--psm 6')
            
            # ================= DEBUG LOG DÀNH CHO ẢNH =================
            print(f"\n[🔍 DEBUG OCR] KẾT QUẢ TỪ FILE ẢNH:")
            print("-" * 50)
            print(text.strip() if text.strip() else "[KHÔNG ĐỌC ĐƯỢC CHỮ NÀO]")
            print("-" * 50 + "\n")
            # ==========================================================
            
            extracted_text += text
            
        print(f"[{OS_NAME}] OCR Hoàn tất!")
        return extracted_text.strip()
        
    except Exception as e:
        print(f"\n[LỖI OCR] Quá trình xử lý thất bại trên {OS_NAME}: {str(e)}")
        raise e