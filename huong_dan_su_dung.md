# Hướng dẫn sử dụng và cài đặt chương trình Code Analyzer

## 1. Giới thiệu
Chương trình Code Analyzer giúp phân tích mã nguồn (C++, Python, JavaScript, Java, ...) để phát hiện lỗi, vấn đề bảo mật, hiệu suất và cải thiện chất lượng mã. Công cụ sử dụng mô hình ngôn ngữ lớn Qwen để phân tích tự động.

---

## 2. Cài đặt chương trình

### 2.1. Yêu cầu hệ thống
- Python >= 3.8
- pip
- (Windows) Visual C++ Build Tools (nếu gặp lỗi khi cài đặt một số package)

### 2.2. Các bước cài đặt

**Bước 1:** Clone repository về máy:
```powershell
git clone https://github.com/VanAnh-13/code_evaluator.git
cd code_evaluator
```

**Bước 2:** Cài đặt các thư viện phụ thuộc (khuyến nghị):
```powershell
python install_dependencies.py
```
Nếu gặp lỗi, có thể thử:
```powershell
pip install -r requirements.txt
```

**Bước 3:** Kiểm tra cài đặt:
```powershell
python test_imports.py
```

---

## 3. Cách sử dụng

### 3.1. Phân tích mã nguồn qua dòng lệnh

- Phân tích một tệp:
```powershell
python code_analyzer.py path\to\your\file.cpp
```
- Phân tích nhiều tệp:
```powershell
python code_analyzer.py file1.cpp file2.py file3.js
```
- Lưu kết quả JSON:
```powershell
python code_analyzer.py file.cpp --output results_dir
```
- Lưu báo cáo Markdown:
```powershell
python code_analyzer.py file.cpp --report reports_dir
```
- Sinh đề xuất sửa lỗi:
```powershell
python code_analyzer.py file.cpp --fix
```

### 3.2. Sử dụng giao diện web

- Chạy server web (Windows):
```powershell
run_web.bat
```
Hoặc:
```powershell
python run_web.py
```
- Truy cập: http://localhost:5000
- Tải lên tệp mã nguồn và nhấn "Analyze" để xem kết quả.

---

## 4. Input/Output

### Input
- Tệp mã nguồn: .cpp, .py, .js, .java, ...
- Có thể upload qua web hoặc chỉ định qua dòng lệnh.

### Output
- Báo cáo phân tích dạng JSON (máy đọc) hoặc Markdown (dễ đọc).
- Đề xuất sửa lỗi (nếu có).
- Hiển thị trực tiếp trên web hoặc lưu ra file.

---

## 5. Mô hình sử dụng
- Sử dụng mô hình Qwen (ví dụ: Qwen/Qwen-7B-Chat, Qwen/Qwen-14B-Chat) từ HuggingFace Transformers.
- Có thể chỉ định model qua tham số `--model`.

---

## 6. Thuật toán và quy trình phân tích
1. **Phát hiện ngôn ngữ:** Dựa vào phần mở rộng tệp.
2. **Kiểm tra cú pháp:** Sử dụng g++ (C++), pylint (Python), eslint (JS), ...
3. **Tạo prompt và gửi mã nguồn cho mô hình Qwen để phân tích sâu:**
   - Phát hiện lỗi logic, quản lý bộ nhớ, bảo mật, hiệu suất, style.
4. **Xử lý kết quả trả về:** Phân loại, đánh giá mức độ nghiêm trọng, sinh đề xuất sửa lỗi.
5. **Sinh báo cáo:** JSON/Markdown.
6. **Caching:** Lưu kết quả để tăng tốc cho lần phân tích lại.

---

## 7. Code mẫu

### Ví dụ C++
**Input:**
```cpp
#include <iostream>
void foo(int* p) {
    *p = 10;
}
int main() {
    int* p = nullptr;
    foo(p);
    return 0;
}
```
**Output (trích đoạn):**
```
# Code Analysis Report
File: example.cpp
Language: cpp

Total issues found: 2

## Bugs and Logical Errors (1)
- Line 4 (high): Potential null pointer dereference
  Recommendation: Add null check before dereferencing pointer

## Memory Management Issues (1)
- Line 7 (critical): Memory leak: allocated memory not freed
  Recommendation: Add delete[] or use smart pointers
```

### Ví dụ Python
**Input:**
```python
def avg(nums):
    return sum(nums) / len(nums)
print(avg([]))
```
**Output (trích đoạn):**
```
# Code Analysis Report
File: example.py
Language: python

Total issues found: 1

## Bugs and Logical Errors (1)
- Line 2 (high): Potential division by zero error
  Recommendation: Add a check to ensure list is not empty before division
```

---

## 8. Một số lưu ý khi cài đặt
- Nếu gặp lỗi khi cài đặt package (đặc biệt là sentencepiece, modelscope), hãy cài đặt Visual C++ Build Tools (Windows) hoặc build-essential (Linux).
- Có thể chạy lại script `install_dependencies.py` sau khi cài build tools.
- Nếu vẫn lỗi, thử cài riêng từng package hoặc tham khảo hướng dẫn chi tiết trong README.md.

---

## 9. Tham khảo thêm
- Xem file `README.md` và `overview_of_implementation_process.md` trong thư mục dự án để biết chi tiết hơn.
