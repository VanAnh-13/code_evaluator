# BÁO CÁO KHOA HỌC

## 1. Giới thiệu

Trong bối cảnh phát triển phần mềm ngày càng phức tạp và nhanh chóng, việc đảm bảo chất lượng mã nguồn không chỉ là một yêu cầu kỹ thuật mà còn là yếu tố then chốt quyết định sự thành công của dự án. Mã nguồn chất lượng cao giúp giảm thiểu lỗi, tăng cường tính bảo mật, cải thiện hiệu suất và làm cho việc bảo trì, mở rộng trở nên dễ dàng hơn. Các phương pháp phân tích mã nguồn truyền thống, như phân tích tĩnh dựa trên quy tắc (rule-based static analysis) hay kiểm thử đơn vị, tuy hữu ích nhưng thường gặp hạn chế trong việc phát hiện các lỗi logic phức tạp, các vấn đề tiềm ẩn về hiệu suất hoặc các lỗ hổng bảo mật tinh vi đòi hỏi sự hiểu biết sâu về ngữ cảnh và ngữ nghĩa của mã.

Sự trỗi dậy của các Mô hình Ngôn ngữ Lớn (LLMs) đã mở ra những hướng tiếp cận mới đầy hứa hẹn cho nhiều lĩnh vực, bao gồm cả phân tích và đánh giá mã nguồn. Với khả năng hiểu và sinh ngôn ngữ tự nhiên cũng như mã nguồn ở mức độ cao, LLMs có tiềm năng vượt qua những hạn chế của các công cụ truyền thống bằng cách cung cấp các phân tích sâu hơn, nhận diện được các mẫu lỗi trừu tượng và đưa ra các đề xuất cải thiện mang tính ngữ cảnh cao.

Bài báo này trình bày về hệ thống "Code Analyzer" – một công cụ tự động được phát triển để phân tích mã nguồn đa ngôn ngữ (bao gồm C++, Python, JavaScript, Java, và các ngôn ngữ phổ biến khác). Hệ thống tận dụng sức mạnh của mô hình ngôn ngữ lớn Qwen, một LLM tiên tiến, để tự động phát hiện các loại lỗi đa dạng, các vấn đề liên quan đến bảo mật, hiệu suất và phong cách viết mã. Mục tiêu của Code Analyzer không chỉ dừng lại ở việc phát hiện lỗi mà còn cung cấp các giải thích chi tiết và đề xuất các phương án cải tiến mã nguồn, qua đó hỗ trợ các nhà phát triển nâng cao chất lượng sản phẩm phần mềm của họ.

## 2. Mục tiêu nghiên cứu

- Xây dựng hệ thống phân tích mã nguồn tự động, hỗ trợ nhiều ngôn ngữ lập trình.
- Ứng dụng mô hình ngôn ngữ lớn (Qwen) để phát hiện lỗi logic, bảo mật, hiệu suất và style code.
- Đưa ra báo cáo phân tích chi tiết và đề xuất sửa lỗi tự động.

## 3. Phương pháp và thuật toán

### 3.1. Quy trình phân tích
Quy trình phân tích mã nguồn của hệ thống Code Analyzer được thiết kế một cách có hệ thống để đảm bảo tính toàn diện và hiệu quả, bao gồm các bước chính sau:

1.  **Phát hiện ngôn ngữ (Language Detection):**
    *   Hệ thống tự động xác định ngôn ngữ lập trình của tệp đầu vào dựa trên phần mở rộng của tệp (ví dụ: `.cpp` cho C++, `.py` cho Python, `.java` cho Java).
    *   Điều này cho phép áp dụng các quy tắc phân tích và công cụ kiểm tra cú pháp đặc thù cho từng ngôn ngữ.

2.  **Kiểm tra cú pháp (Syntax Checking):**
    *   Trước khi phân tích sâu bằng LLM, mã nguồn được kiểm tra cú pháp sơ bộ bằng các công cụ chuyên dụng cho từng ngôn ngữ.
    *   Ví dụ: `g++ -fsyntax-only` cho C++, `pylint` hoặc `flake8` cho Python, `eslint` cho JavaScript, `javac` cho Java.
    *   Bước này giúp phát hiện sớm các lỗi cú pháp cơ bản, tránh việc gửi mã không hợp lệ đến LLM, tiết kiệm tài nguyên và thời gian xử lý. Các lỗi cú pháp nghiêm trọng có thể ngăn cản quá trình phân tích sâu hơn.

3.  **Tạo prompt và tương tác với mô hình Qwen (Prompt Generation and LLM Interaction):**
    *   Đây là bước cốt lõi của quy trình. Một prompt (câu lệnh đầu vào) được xây dựng cẩn thận, bao gồm toàn bộ mã nguồn cần phân tích và các chỉ dẫn cụ thể cho mô hình LLM.
    *   Prompt yêu cầu mô hình Qwen thực hiện phân tích đa chiều: phát hiện lỗi logic tiềm ẩn, các vấn đề về quản lý bộ nhớ (như rò rỉ bộ nhớ, truy cập con trỏ null), lỗ hổng bảo mật (SQL injection, buffer overflow), các điểm nghẽn về hiệu suất, và các vi phạm quy tắc về style code.
    *   Hệ thống gửi prompt này đến mô hình Qwen (ví dụ: Qwen-7B-Chat, Qwen-14B-Chat) thông qua API hoặc thư viện `transformers`.

4.  **Xử lý và phân loại kết quả (Result Processing and Categorization):**
    *   Kết quả trả về từ mô hình Qwen (thường ở dạng văn bản) được phân tích và cấu trúc hóa.
    *   Các vấn đề được phát hiện sẽ được phân loại vào các hạng mục cụ thể như: Lỗi logic (Bugs and Logical Errors), Vấn đề quản lý bộ nhớ (Memory Management Issues), Lỗ hổng bảo mật (Security Vulnerabilities), Vấn đề hiệu suất (Performance Issues), và Gợi ý về style code (Code Style and Readability).
    *   Mỗi vấn đề được đánh giá mức độ nghiêm trọng (ví dụ: Critical, High, Medium, Low) để người dùng có thể ưu tiên xử lý.
    *   Hệ thống cũng cố gắng sinh ra các đề xuất sửa lỗi (recommendations) cụ thể cho từng vấn đề.

5.  **Sinh báo cáo (Report Generation):**
    *   Kết quả phân tích chi tiết được tổng hợp và trình bày dưới dạng báo cáo.
    *   Hệ thống hỗ trợ xuất báo cáo ở nhiều định dạng:
        *   **JSON:** Định dạng máy đọc, phù hợp cho việc tích hợp với các công cụ khác hoặc xử lý tự động.
        *   **Markdown:** Định dạng dễ đọc cho người dùng, trình bày rõ ràng các vấn đề, vị trí lỗi, và đề xuất sửa lỗi.

6.  **Caching (Kết quả tạm thời):**
    *   Để tối ưu hóa hiệu suất và giảm thời gian chờ cho các lần phân tích lặp lại trên cùng một tệp mã nguồn (nếu không có thay đổi), hệ thống có thể triển khai cơ chế caching.
    *   Kết quả phân tích của một tệp sẽ được lưu trữ tạm thời. Nếu tệp đó được yêu cầu phân tích lại và nội dung không thay đổi, kết quả từ cache sẽ được trả về ngay lập tức.

### 3.2. Mô hình sử dụng
Việc lựa chọn mô hình ngôn ngữ lớn (LLM) là một yếu tố then chốt quyết định đến chất lượng phân tích của hệ thống. Code Analyzer ưu tiên sử dụng các mô hình thuộc dòng Qwen, được phát triển bởi Alibaba Cloud, do khả năng hiểu và sinh mã tốt, cũng như hỗ trợ đa ngôn ngữ hiệu quả.

-   **Các phiên bản Qwen:**
    *   Hệ thống được thiết kế để tương thích với các phiên bản khác nhau của Qwen, phổ biến nhất là `Qwen-7B-Chat` và `Qwen-14B-Chat`. Các con số "7B" và "14B" chỉ số lượng tham số của mô hình (7 tỷ và 14 tỷ), thường thì mô hình có nhiều tham số hơn sẽ có khả năng hiểu sâu hơn nhưng cũng đòi hỏi tài nguyên tính toán lớn hơn.
    *   Phiên bản "Chat" được tối ưu cho các tác vụ đối thoại và hoàn thành chỉ dẫn, rất phù hợp với việc phân tích mã nguồn theo yêu cầu cụ thể trong prompt.
-   **Tích hợp qua HuggingFace Transformers:**
    *   Các mô hình Qwen được truy cập và sử dụng thông qua thư viện `transformers` của HuggingFace, một nền tảng phổ biến cung cấp hàng ngàn mô hình học máy tiền huấn luyện. Điều này giúp việc tích hợp và cập nhật mô hình trở nên dễ dàng.
-   **Khả năng tùy chọn mô hình:**
    *   Hệ thống cho phép người dùng linh hoạt lựa chọn phiên bản mô hình Qwen cụ thể muốn sử dụng thông qua tham số dòng lệnh (ví dụ: `--model Qwen-14B-Chat`). Điều này cho phép người dùng cân bằng giữa chất lượng phân tích và yêu cầu về tài nguyên phần cứng.
    *   Trong tương lai, hệ thống có thể được mở rộng để hỗ trợ các LLM khác ngoài Qwen, tùy thuộc vào sự phát triển của công nghệ và nhu cầu người dùng.
-   **Lý do lựa chọn Qwen:**
    *   **Hiệu suất tốt trên mã nguồn:** Qwen đã chứng minh khả năng tốt trong việc hiểu các cấu trúc mã phức tạp và phát hiện các loại lỗi đa dạng.
    *   **Hỗ trợ đa ngôn ngữ:** Khả năng xử lý nhiều ngôn ngữ lập trình là một ưu điểm lớn.
    *   **Cộng đồng và tài liệu:** Việc có sẵn trên HuggingFace và được hỗ trợ bởi cộng đồng giúp việc triển khai và khắc phục sự cố thuận lợi hơn.

## 4. Hiện thực chương trình

Chương này mô tả chi tiết về cấu trúc và các thành phần mã nguồn chính của hệ thống Code Analyzer.

### 4.1. Cấu trúc thư mục

Dự án được tổ chức với cấu trúc thư mục như sau, đảm bảo tính module và dễ quản lý:

```
code_evaluator/
├── code_analyzer.py            # Module chính xử lý logic phân tích mã
├── cpp_code_analyzer.py        # Module chuyên biệt cho phân tích C++ (ví dụ)
├── install_dependencies.py     # Script cài đặt các thư viện cần thiết
├── requirements.txt            # Danh sách các thư viện Python phụ thuộc
├── run_web.py                  # Script khởi chạy ứng dụng web
├── Dockerfile                  # Cấu hình Docker cho triển khai
├── examples/                   # Chứa các tệp mã nguồn ví dụ
│   ├── example.cpp
│   ├── example.py
│   └── example.js
├── web_app/                    # Module ứng dụng web Flask
│   ├── app.py                  # File chính của ứng dụng Flask
│   ├── static/                 # Chứa các tệp tĩnh (CSS, JS, images)
│   │   ├── css/style.css
│   │   └── js/main.js
│   └── templates/              # Chứa các template HTML
│       ├── base.html
│       ├── index.html
│       └── analysis.html
├── README.md
└── ... (các file khác)
```

### 4.2. Module chính: `code_analyzer.py`

Đây là thành phần trung tâm, chịu trách nhiệm cho toàn bộ quy trình phân tích mã nguồn. Các chức năng chính bao gồm:

*   **Phát hiện ngôn ngữ:** Tự động xác định ngôn ngữ lập trình dựa trên phần mở rộng của tệp.
*   **Kiểm tra cú pháp:** Tích hợp với các linter và trình biên dịch (ví dụ: `g++` cho C++, `pylint` cho Python) để phát hiện lỗi cú pháp cơ bản.
*   **Tương tác với mô hình LLM (Qwen):**
    *   Chuẩn bị prompt đầu vào chứa mã nguồn và yêu cầu phân tích.
    *   Gửi prompt đến mô hình Qwen (thông qua API hoặc thư viện `transformers`).
    *   Nhận và xử lý kết quả trả về từ mô hình.
*   **Phân loại và định dạng kết quả:** Tổ chức các vấn đề được phát hiện thành các hạng mục (lỗi logic, bảo mật, hiệu suất, ...) và mức độ nghiêm trọng.
*   **Đề xuất sửa lỗi:** Dựa trên kết quả từ LLM và các quy tắc được định nghĩa sẵn để đưa ra gợi ý sửa lỗi.
*   **Sinh báo cáo:** Tạo báo cáo phân tích dưới dạng JSON hoặc Markdown.

**Trích đoạn mã nguồn `code_analyzer.py` (minh họa chức năng phân tích chính):**

```python
# ... (imports and helper functions) ...

class CodeAnalyzer:
    def __init__(self, model_name="Qwen/Qwen-7B-Chat", cache_dir=".cache"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.cache_dir = cache_dir
        self.cache = self.load_cache()
        # ... (initialization)

    def load_model(self):
        # Load the Qwen model and tokenizer from HuggingFace
        # Handles potential errors during model loading
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            print(f"[INFO] Model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            # Fallback or exit strategy

    def analyze_code_with_llm(self, code: str, language: str, fix_issues: bool = False) -> Dict[str, Any]:
        # Prepare prompt for the LLM
        prompt = self._build_prompt(code, language, fix_issues)
        
        # Interact with the LLM (Qwen)
        # This is a simplified representation
        try:
            if not self.model or not self.tokenizer:
                return {"error": "Model not loaded."}
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse LLM response to extract issues, recommendations, fixes
            parsed_results = self._parse_llm_response(response_text, language)
            return parsed_results
        except Exception as e:
            return {"error": f"LLM analysis failed: {e}"}

    def analyze_file(self, file_path: str, fix_issues: bool = False) -> Dict[str, Any]:
        # Main function to analyze a single file
        # ... (read file, detect language, check cache) ...
        language = detect_language(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Step 1: Syntax Check
        syntax_errors = check_syntax(code, language)
        results = {
            "file_path": file_path,
            "language": language,
            "syntax_errors": syntax_errors,
            # ... other categories ...
        }

        # Step 2: LLM Analysis (if no critical syntax errors or as configured)
        llm_results = self.analyze_code_with_llm(code, language, fix_issues)
        results.update(llm_results) # Merge results
        
        # ... (save to cache, generate suggested fixes if fix_issues is True) ...
        return results

# ... (other functions like generate_report, save_results, check_syntax, detect_language) ...
```

### 4.3. Giao diện Web: `web_app/app.py`

Để cung cấp trải nghiệm người dùng thân thiện, hệ thống tích hợp một giao diện web sử dụng Flask. Người dùng có thể tải lên tệp mã nguồn, xem kết quả phân tích và lịch sử các lần phân tích.

Các chức năng chính của module web:

*   **Xử lý upload tệp:** Nhận tệp mã nguồn từ người dùng, kiểm tra định dạng và lưu trữ tạm thời.
*   **Gọi module phân tích:** Sử dụng `CodeAnalyzer` từ `code_analyzer.py` để thực hiện phân tích.
*   **Hiển thị kết quả:** Trình bày báo cáo phân tích một cách trực quan trên giao diện web.
*   **Quản lý lịch sử:** Lưu trữ và hiển thị lịch sử các tệp đã được phân tích.

**Trích đoạn mã nguồn `web_app/app.py` (minh họa route xử lý phân tích):**

```python
# ... (imports: Flask, render_template, request, etc.) ...
# Add parent directory to path to import code_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_analyzer import CodeAnalyzer, generate_report # Import the main analyzer

app = Flask(__name__)
# ... (app.config settings: SECRET_KEY, UPLOAD_FOLDER, ALLOWED_EXTENSIONS) ...

# Initialize the analyzer (ideally once)
analyzer = CodeAnalyzer()
analyzer.load_model() # Load LLM model when app starts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # ... (file validation and saving logic) ...
    if file and allowed_file(file.filename):
        # ... (save file, get unique path) ...
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        # ... (store file_id in session) ...
        return redirect(url_for('analyze', file_id=file_id))
    # ... (error handling) ...

@app.route('/analyze/<file_id>')
def analyze(file_id):
    # ... (retrieve file_info from session using file_id) ...
    file_info = next((f for f in session.get('files', []) if f['id'] == file_id), None)
    if not file_info:
        flash('File not found', 'error')
        return redirect(url_for('index'))

    try:
        # Analyze the file using the core analyzer module
        results = analyzer.analyze_file(file_info['path'])
        report_md = generate_report(results) # Generate Markdown report
        
        # Convert Markdown to HTML for display (optional, can use a library)
        # For simplicity, passing Markdown directly or a parsed structure
        
        file_info['results'] = results # Store full results
        session.modified = True
        
        return render_template('analysis.html', 
                               file_info=file_info, 
                               report_md=report_md, # Pass report to template
                               results_json=json.dumps(results, indent=2)) # Pass JSON for detailed view
    except Exception as e:
        flash(f'Error during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

# ... (routes for history, clear_history, error handlers) ...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
```

### 4.4. Các thành phần khác

*   **`install_dependencies.py`:** Script Python tự động hóa việc cài đặt các thư viện cần thiết được liệt kê trong `requirements.txt`.
*   **`requirements.txt`:** Tệp văn bản liệt kê tất cả các thư viện Python mà dự án phụ thuộc (ví dụ: `Flask`, `torch`, `transformers`, `pylint`).
*   **`run_web.py`:** Một script đơn giản để khởi chạy ứng dụng web Flask, thường chỉ gọi `app.run()` từ `web_app/app.py`.
*   **`Dockerfile`:** Cung cấp các chỉ dẫn để xây dựng một Docker image cho ứng dụng, giúp việc triển khai trở nên dễ dàng và nhất quán trên các môi trường khác nhau.

## 5. Cài đặt và triển khai

### 5.1. Yêu cầu hệ thống
- Python >= 3.8
- pip
- (Windows) Visual C++ Build Tools (nếu gặp lỗi khi cài đặt một số package)

### 5.2. Cài đặt
- Clone repository:
```powershell
git clone https://github.com/VanAnh-13/code_evaluator.git
cd code_evaluator
```
- Cài đặt phụ thuộc:
```powershell
python install_dependencies.py
```
- Kiểm tra cài đặt:
```powershell
python test_imports.py
```

### 5.3. Triển khai (Deployment)

Ngoài việc chạy trực tiếp trên máy local, chương trình có thể được triển khai bằng Docker để đảm bảo tính nhất quán môi trường và dễ dàng phân phối.

**Dockerfile mẫu:**

```dockerfile
# Sử dụng base image Python
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt và cài đặt thư viện
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của ứng dụng vào thư mục làm việc
COPY . .

# Thiết lập biến môi trường (nếu cần)
# ENV MODEL_NAME=Qwen/Qwen-7B-Chat

# Mở port cho ứng dụng web (nếu có)
EXPOSE 5000

# Lệnh để chạy ứng dụng khi container khởi động
# Ví dụ cho ứng dụng web:
CMD ["python", "run_web.py"]
# Hoặc cho công cụ dòng lệnh (cần điều chỉnh cho phù hợp):
# CMD ["python", "code_analyzer.py"]
```

**Các bước build và run Docker image:**

1.  **Build Docker image:**
    ```powershell
    docker build -t code-analyzer-app .
    ```

2.  **Run Docker container (ví dụ cho web app):**
    ```powershell
    docker run -p 5000:5000 code-analyzer-app
    ```
    Sau đó truy cập `http://localhost:5000` trên trình duyệt.

## 6. Sử dụng hệ thống

### 6.1. Qua dòng lệnh
- Phân tích một hoặc nhiều tệp:
```powershell
python code_analyzer.py file1.cpp file2.py
```
- Lưu kết quả JSON:
```powershell
python code_analyzer.py file.cpp --output results_dir
```
- Sinh báo cáo Markdown:
```powershell
python code_analyzer.py file.cpp --report reports_dir
```
- Đề xuất sửa lỗi:
```powershell
python code_analyzer.py file.cpp --fix
```

### 6.2. Qua giao diện web
- Chạy server:
```powershell
python run_web.py
```
- Truy cập: http://localhost:5000

## 7. Kết quả thực nghiệm

### 7.1. Ví dụ C++
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
**Output:**
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

**Input (Ví dụ 2):**
```cpp
#include <iostream>
#include <vector>

void processVector(std::vector<int>& data) {
    // Potential out-of-bounds access
    if (data.size() > 5) { // Should be data.size() >= 5 or check specific index
        std::cout << "Accessing element 5: " << data[5] << std::endl;
    }
}

int main() {
    std::vector<int> myVector = {1, 2, 3, 4}; // Only 4 elements
    processVector(myVector);

    int uninitializedValue;
    // Using uninitialized variable
    std::cout << "Uninitialized value: " << uninitializedValue << std::endl;

    return 0;
}
```
**Output (trích đoạn - Ví dụ 2):**
```
# Code Analysis Report
File: example2.cpp
Language: cpp

Total issues found: 2

## Bugs and Logical Errors (2)
- Line 7 (medium): Potential array out-of-bounds access. The vector 'data' might have fewer than 6 elements, but element at index 5 is accessed.
  Recommendation: Ensure 'data.size()' is at least 6 before accessing 'data[5]', or adjust the condition.
- Line 15 (medium): Variable 'uninitializedValue' is used before it has been initialized.
  Recommendation: Initialize 'uninitializedValue' with a default value before its first use.
```

### 7.2. Ví dụ Python
**Input:**
```python
def avg(nums):
    return sum(nums) / len(nums)
print(avg([]))
```
**Output:**
```
# Code Analysis Report
File: example.py
Language: python

Total issues found: 1

## Bugs and Logical Errors (1)
- Line 2 (high): Potential division by zero error
  Recommendation: Add a check to ensure list is not empty before division
```

### 7.3. Ví dụ Java
**Input:**
```java
// File: JavaExample.java
public class JavaExample {
    public static void main(String[] args) {
        String s = null;
        System.out.println(s.length()); // Potential NullPointerException

        int[] arr = new int[5];
        System.out.println(arr[5]); // Potential ArrayIndexOutOfBoundsException

        // Example of a resource leak (not automatically closed)
        try {
            java.io.FileInputStream fis = new java.io.FileInputStream("file.txt");
            // fis.read(); // Operation on fis
        } catch (java.io.IOException e) {
            e.printStackTrace();
        }
        // fis is not closed here if an exception occurs or normally
    }

    public void unusedMethod() {
        // This method is not used
        System.out.println("This method is unused.");
    }

    public void inefficientLoop(java.util.List<String> list) {
        String result = "";
        if (list != null) {
            for (int i = 0; i < list.size(); i++) {
                result += list.get(i); // Inefficient string concatenation in loop
            }
        }
        System.out.println(result);
    }
}
```
**Output (trích đoạn):**
```
# Code Analysis Report
File: JavaExample.java
Language: java

Total issues found: 4

## Bugs and Logical Errors (2)
- Line 5 (critical): Potential NullPointerException. Variable 's' is null when 's.length()' is called.
  Recommendation: Ensure 's' is initialized or check for null before calling methods on it.
- Line 8 (high): Potential ArrayIndexOutOfBoundsException. Array 'arr' has size 5 (indices 0-4), but index 5 is accessed.
  Recommendation: Check array bounds before accessing elements. Loop from 0 to arr.length - 1.

## Memory Management Issues (1)
- Line 11 (medium): Potential resource leak. FileInputStream 'fis' is not guaranteed to be closed, especially if an IOException occurs.
  Recommendation: Use a try-with-resources statement to ensure 'fis' is closed automatically. E.g., 'try (java.io.FileInputStream fis = new java.io.FileInputStream("file.txt")) { ... }'

## Performance Issues (1)
- Line 25 (medium): Inefficient string concatenation in a loop using '+='.
  Recommendation: Use 'StringBuilder' for more efficient string concatenation in loops. E.g., 'StringBuilder sb = new StringBuilder(); for (String item : list) { sb.append(item); } result = sb.toString();'

## Code Style and Readability (1)
- Line 19 (low): Method 'unusedMethod' is declared but never used.
  Recommendation: Remove unused methods to improve code clarity and reduce dead code, or add a @SuppressWarnings("unused") annotation if intentionally unused.
```

## 8. Đánh giá và thảo luận

Việc ứng dụng Mô hình Ngôn ngữ Lớn (LLM) như Qwen vào phân tích mã nguồn mang lại nhiều ưu điểm vượt trội so với các phương pháp truyền thống, nhưng cũng đi kèm với những thách thức và hạn chế nhất định.

**Ưu điểm:**
*   **Khả năng hiểu ngữ nghĩa sâu:** LLMs có khả năng hiểu ngữ cảnh và ý định đằng sau mã nguồn, giúp phát hiện các lỗi logic tinh vi hoặc các vấn đề về thiết kế mà các công cụ dựa trên quy tắc thường bỏ sót.
*   **Phân tích đa diện:** Hệ thống không chỉ giới hạn ở việc tìm lỗi cú pháp hay các mẫu lỗi cố định mà còn có thể đánh giá về hiệu suất, bảo mật, khả năng đọc hiểu, và đưa ra các đề xuất cải thiện phong cách viết mã.
*   **Linh hoạt với nhiều ngôn ngữ:** Với việc huấn luyện trên kho dữ liệu mã nguồn khổng lồ đa ngôn ngữ, LLMs như Qwen có thể dễ dàng thích ứng với việc phân tích nhiều ngôn ngữ lập trình khác nhau mà không cần xây dựng bộ quy tắc riêng cho từng ngôn ngữ.
*   **Đề xuất sửa lỗi mang tính xây dựng:** Thay vì chỉ thông báo lỗi, hệ thống có thể cung cấp các giải thích chi tiết và các đoạn mã gợi ý để khắc phục vấn đề, giúp nhà phát triển học hỏi và cải thiện kỹ năng.

**Hạn chế và Thách thức:**
*   **Phụ thuộc vào chất lượng mô hình và dữ liệu huấn luyện:**
    *   Độ chính xác và hiệu quả của việc phân tích phụ thuộc lớn vào chất lượng của mô hình LLM được sử dụng và sự đa dạng, phong phú của dữ liệu mà nó đã được huấn luyện. Mô hình có thể "ảo giác" (hallucinate) hoặc đưa ra các phân tích sai lệch nếu gặp phải các mẫu mã quá mới lạ hoặc nằm ngoài phạm vi kiến thức của nó.
    *   Thiên kiến (bias) trong dữ liệu huấn luyện có thể dẫn đến việc mô hình ưu tiên một số phong cách viết mã nhất định hoặc bỏ qua một số loại lỗi cụ thể.
*   **Yêu cầu tài nguyên tính toán:**
    *   Việc chạy các mô hình LLM lớn đòi hỏi tài nguyên phần cứng đáng kể (GPU mạnh, bộ nhớ lớn), đặc biệt là cho việc phân tích các dự án mã nguồn lớn hoặc khi cần tốc độ phản hồi nhanh. Điều này có thể là rào cản cho các nhà phát triển cá nhân hoặc các nhóm nhỏ.
    *   Nếu sử dụng API của các nhà cung cấp LLM, chi phí có thể tăng lên đáng kể tùy thuộc vào khối lượng mã được phân tích.
*   **Thời gian phân tích:**
    *   Đối với các đoạn mã dài hoặc phức tạp, thời gian để LLM xử lý và đưa ra phản hồi có thể lâu hơn so với các công cụ phân tích tĩnh truyền thống.
*   **Độ tin cậy và kiểm chứng kết quả:**
    *   Mặc dù LLMs ngày càng trở nên mạnh mẽ, kết quả phân tích đôi khi vẫn cần được con người kiểm chứng lại, đặc biệt là đối với các cảnh báo về bảo mật hoặc các đề xuất thay đổi cấu trúc mã lớn. Việc tự động áp dụng các sửa lỗi do LLM đề xuất mà không có sự giám sát có thể tiềm ẩn rủi ro.
*   **Khó khăn trong việc diễn giải "hộp đen":**
    *   Việc hiểu rõ tại sao LLM đưa ra một phân tích hoặc đề xuất cụ thể đôi khi rất khó khăn do tính chất "hộp đen" của các mô hình này. Điều này gây khó khăn cho việc gỡ lỗi và cải thiện mô hình một cách có hệ thống.

**Hướng phát triển tương lai:**
Để khắc phục các hạn chế và nâng cao hơn nữa hiệu quả của hệ thống, một số hướng phát triển tiềm năng bao gồm:

*   **Fine-tuning mô hình chuyên biệt:** Huấn luyện bổ sung (fine-tuning) mô hình Qwen (hoặc các LLM khác) trên các tập dữ liệu mã nguồn chuyên biệt cho từng ngôn ngữ hoặc từng loại lỗi cụ thể (ví dụ: lỗ hổng bảo mật web, lỗi trong hệ thống nhúng) để tăng độ chính xác và giảm thiểu "ảo giác".
*   **Tích hợp với các công cụ phân tích truyền thống:** Kết hợp sức mạnh của LLM với các công cụ phân tích tĩnh dựa trên quy tắc. LLM có thể được sử dụng để xác minh hoặc làm giàu kết quả từ các công cụ này, hoặc ngược lại, kết quả từ các công cụ truyền thống có thể được dùng làm thông tin đầu vào bổ sung cho LLM.
*   **Phân tích theo ngữ cảnh dự án:** Mở rộng khả năng phân tích để xem xét toàn bộ ngữ cảnh của dự án (ví dụ: các phụ thuộc, luồng dữ liệu giữa các module) thay vì chỉ phân tích từng tệp riêng lẻ.
*   **Tối ưu hóa hiệu suất và tài nguyên:** Nghiên cứu các kỹ thuật chưng cất mô hình (model distillation), lượng tử hóa (quantization) để giảm kích thước mô hình và yêu cầu tài nguyên mà vẫn giữ được chất lượng phân tích chấp nhận được.
*   **Cải thiện khả năng tương tác và diễn giải:** Phát triển các giao diện hoặc cơ chế cho phép người dùng tương tác sâu hơn với quá trình phân tích, ví dụ như đặt câu hỏi cụ thể về một đoạn mã hoặc yêu cầu giải thích chi tiết hơn về một đề xuất của LLM.
*   **Hỗ trợ phân tích mã theo thời gian thực (Real-time Analysis):** Tích hợp vào IDE để cung cấp phản hồi và gợi ý ngay trong quá trình viết mã.

## 9. Kết luận

Bài báo này đã trình bày chi tiết về hệ thống "Code Analyzer", một công cụ phân tích mã nguồn tự động đa ngôn ngữ dựa trên nền tảng Mô hình Ngôn ngữ Lớn Qwen. Hệ thống đã chứng minh được tiềm năng to lớn của việc ứng dụng LLMs trong việc nâng cao chất lượng phần mềm, không chỉ dừng lại ở việc phát hiện lỗi cú pháp cơ bản mà còn đi sâu vào phân tích ngữ nghĩa, logic, bảo mật và hiệu suất của mã nguồn. 

Các đóng góp chính của nghiên cứu bao gồm:
1.  **Xây dựng một quy trình phân tích mã toàn diện:** Từ việc phát hiện ngôn ngữ, kiểm tra cú pháp, đến việc tạo prompt thông minh cho LLM và xử lý kết quả phân tích một cách có cấu trúc.
2.  **Ứng dụng thành công mô hình Qwen:** Khai thác khả năng của LLM để thực hiện các tác vụ phân tích phức tạp, đưa ra các nhận định sâu sắc và các đề xuất cải thiện mã nguồn hữu ích.
3.  **Cung cấp giao diện người dùng đa dạng:** Hỗ trợ cả giao diện dòng lệnh cho các quy trình tự động hóa và giao diện web trực quan cho người dùng cuối.
4.  **Đánh giá khách quan những ưu điểm và hạn chế:** Nhận diện rõ ràng những lợi ích mà LLM mang lại cũng như các thách thức cần giải quyết để công nghệ này thực sự trở nên phổ biến và đáng tin cậy trong thực tiễn phát triển phần mềm.

Hệ thống Code Analyzer không chỉ là một công cụ hỗ trợ lập trình viên trong việc phát hiện và sửa lỗi nhanh chóng mà còn đóng vai trò như một người đồng hành thông minh, giúp cải thiện kỹ năng viết mã và tuân thủ các tiêu chuẩn chất lượng cao. Mặc dù vẫn còn những hạn chế và không gian để cải tiến, hướng tiếp cận sử dụng LLM cho phân tích mã nguồn hứa hẹn sẽ tiếp tục phát triển mạnh mẽ, đóng góp vào việc xây dựng các sản phẩm phần mềm ngày càng an toàn, hiệu quả và dễ bảo trì hơn. Trong tương lai, với sự phát triển của các mô hình AI mạnh mẽ hơn và các kỹ thuật fine-tuning chuyên biệt, các công cụ như Code Analyzer sẽ ngày càng trở nên thông minh và không thể thiếu trong bộ công cụ của mọi nhà phát triển phần mềm.

## 10. Tài liệu tham khảo

[1] A. Vaswani et al., "Attention is All You Need," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017, pp. 5998-6008.

[2] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*, 2019, pp. 4171-4186.

[3] Qwen Team, Alibaba Cloud. "Qwen Technical Report." arXiv preprint arXiv:2309.16609, 2023. [Online]. Available: https://arxiv.org/abs/2309.16609

[4] T. Chen et al., "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis," arXiv preprint arXiv:2203.13474, 2022. [Online]. Available: https://arxiv.org/abs/2203.13474

[5] S. McConnell, *Code Complete: A Practical Handbook of Software Construction*, 2nd ed. Redmond, WA: Microsoft Press, 2004.

[6] M. Fowler, *Refactoring: Improving the Design of Existing Code*, 2nd ed. Boston, MA: Addison-Wesley Professional, 2018.

[7] Hugging Face, "Transformers: State-of-the-art Natural Language Processing." [Online]. Available: https://huggingface.co/transformers/

[8] P. J. Deitel and H. M. Deitel, *C++ How to Program*, 10th ed. Pearson Education, 2016.

[9] J. Bloch, *Effective Java*, 3rd ed. Addison-Wesley Professional, 2018.

[10] Python Software Foundation, "Python Language Reference." [Online]. Available: https://docs.python.org/3/reference/index.html
