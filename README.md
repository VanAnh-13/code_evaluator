# Code Analyzer using Qwen

This tool analyzes code in multiple programming languages for potential issues, code quality, and security vulnerabilities using the Qwen large language model.

## Features

- Supports multiple programming languages (C++, Python, JavaScript, Java, and more)
- Detects potential bugs and logical errors
- Identifies memory and resource management issues
- Finds security vulnerabilities
- Highlights performance issues
- Suggests code style and readability improvements
- Generates language-specific suggested fixes for identified issues
- Supports analyzing multiple files
- Caches analysis results for improved performance
- Provides detailed reports in both JSON and Markdown formats
- Web interface for uploading and analyzing code files
- ChatGPT-like UI for viewing analysis results

## Installation

### Using pip

```bash
# Clone the repository
https://github.com/VanAnh-13/code_evaluator.git
cd cpp-code-analyzer

# Install dependencies (Method 1 - Recommended)
python install_dependencies.py

# OR Install dependencies manually (Method 2)
pip install -r requirements.txt
```

### Using Docker

```bash
# Build the Docker image
docker build -t code-analyzer .

# Run the analyzer using Docker
docker run -v $(pwd):/data code-analyzer /data/your_file.cpp

# Analyze Python files
docker run -v $(pwd):/data code-analyzer /data/your_script.py

# Analyze JavaScript files
docker run -v $(pwd):/data code-analyzer /data/your_script.js
```

## Usage

```bash
# Basic usage with a single file
python code_analyzer.py path/to/your/file.cpp

# Analyze Python files
python code_analyzer.py path/to/your/script.py

# Analyze JavaScript files
python code_analyzer.py path/to/your/script.js

# Analyze multiple files of different languages
python code_analyzer.py file1.cpp file2.py file3.js

# Save results to a directory (creates JSON files for each analyzed file)
python code_analyzer.py path/to/your/file.cpp --output results_dir

# Save human-readable reports to a directory
python code_analyzer.py path/to/your/file.py --report reports_dir

# Generate suggested fixes for identified issues
python code_analyzer.py path/to/your/file.js --fix

# Disable caching of analysis results
python code_analyzer.py path/to/your/file.cpp --no-cache

# Enable verbose output
python code_analyzer.py path/to/your/file.py --verbose

# Specify a different Qwen model
python code_analyzer.py path/to/your/file.js --model Qwen/Qwen-14B-Chat
```

## Requirements

- Python 3.8+
- transformers==4.38.2
- torch==2.2.2+cpu (CPU-only version)
- modelscope==1.9.5
- sentencepiece==0.1.99
- colorama==0.4.6
- tqdm==4.66.1
- numpy==1.24.4
- flask==2.3.3
- werkzeug==2.3.7
- flask-wtf==1.2.1
- wtforms==3.0.1
- python-dotenv==1.0.0

### Language-Specific Requirements

- g++ (for C++ syntax checking)
- pylint (for Python syntax checking)
- node.js and eslint (for JavaScript syntax checking, optional)
- javac (for Java syntax checking, optional)
- dotnet (for C# syntax checking, optional)

## Examples

### C++ Example

Input C++ file (`example.cpp`):

```cpp
#include <iostream>
#include <cstring>

void process_data(char* data, size_t size) {
    char* buffer = new char[size];
    memcpy(buffer, data, size);

    // Process buffer
    for (size_t i = 0; i < size; i++) {
        buffer[i] = toupper(buffer[i]);
    }

    // Return processed data
    memcpy(data, buffer, size);

    // Forgot to delete buffer
}

int main() {
    char data[] = "hello world";
    process_data(data, strlen(data));
    std::cout << data << std::endl;
    return 0;
}
```

Output report:

```
# Code Analysis Report
File: example.cpp
Language: cpp

Total issues found: 5

## Bugs and Logical Errors (1)
- Line 15 (high): Potential null pointer dereference
  Recommendation: Add null check before dereferencing pointer

## Memory Management Issues (1)
- Line 23 (critical): Memory leak: allocated memory not freed
  Recommendation: Add delete[] or use smart pointers

## Security Vulnerabilities (1)
- Line 42 (medium): Buffer overflow risk in strcpy
  Recommendation: Use strncpy or std::string instead

## Performance Issues (1)
- Line 57 (low): Inefficient loop implementation
  Recommendation: Consider using std::transform or range-based for loop

## Code Style and Readability (1)
- Line 10 (info): Inconsistent naming convention
  Recommendation: Follow a consistent naming style (e.g., camelCase or snake_case)
```

### Python Example

Input Python file (`example.py`):

```python
def calculate_average(numbers):
    total = 0
    count = 0

    for num in numbers:
        total += num
        count += 1

    # Potential division by zero if numbers is empty
    return total / count

def main():
    # Undefined variable used
    print(user_input)

    # Inefficient list creation
    result = []
    for i in range(1000):
        result.append(i * i)

    # Resource not properly closed
    f = open("data.txt", "r")
    content = f.read()
    print(content)

    # Calculating average of empty list will cause error
    print(calculate_average([]))

if __name__ == "__main__":
    main()
```

Output report:

```
# Code Analysis Report
File: example.py
Language: python

Total issues found: 4

## Syntax Errors (1)
- Line 13 (high): Syntax issue: Undefined variable 'user_input'
  Recommendation: Define variable 'user_input' before using it

## Bugs and Logical Errors (1)
- Line 9 (high): Potential division by zero error
  Recommendation: Add a check to ensure count is not zero before division

## Resource Management Issues (1)
- Line 21 (medium): File resource not properly closed
  Recommendation: Use 'with open()' statement or call f.close()

## Performance Issues (1)
- Line 16 (low): Inefficient list creation
  Recommendation: Use list comprehension: result = [i * i for i in range(1000)]
```

## Full Implementation

The full implementation uses the Qwen model to analyze code in multiple programming languages. It loads the model using the transformers library and generates detailed analysis based on the code content. The analyzer automatically detects the programming language based on the file extension and applies language-specific analysis techniques.

## Optimizations

This tool has been optimized for performance and efficiency:

### Code Optimizations
- **Caching mechanism**: Analysis results are cached based on file modification time to avoid redundant processing
- **Improved error handling**: Specific error types with descriptive messages
- **Multiple file support**: Process multiple C++ files in a single run
- **Memory efficiency**: Avoid reading files multiple times
- **Progress reporting**: Verbose mode for detailed progress information

### Docker Optimizations
- **Multi-stage build**: Reduces final image size by separating build and runtime environments
- **Layer caching**: Optimized layer structure for faster builds
- **Security enhancements**: Runs as non-root user
- **Environment variables**: Configured for optimal Python performance
- **Pinned dependencies**: Exact versions specified for reproducibility

## Web Interface

The Code Analyzer includes a web interface with a ChatGPT-like UI for uploading and analyzing code files in multiple programming languages.

### Running the Web Server

#### Windows Users

```bash
# Method 1: Using the batch file (easiest)
run_web.bat

# Method 2: Using the Python script
python run_web.py

# Method 3: Directly running the Flask app
cd web_app
python app.py
```

#### Linux/Mac Users

```bash
# Method 1: Using the shell script (easiest)
chmod +x run_web.sh  # Make the script executable (first time only)
./run_web.sh

# Method 2: Using the Python script
python3 run_web.py

# Method 3: Directly running the Flask app
cd web_app
python3 app.py
```

The web server will start on http://localhost:5000 by default. You can access the web interface by opening this URL in your browser.

#### Troubleshooting

If you encounter a "ModuleNotFoundError: No module named 'flask'" error, you need to install the required dependencies:

```bash
# Windows
python install_dependencies.py

# Linux/Mac
python3 install_dependencies.py
```

### Using the Web Interface

1. Open your browser and navigate to http://localhost:5000
2. Click the "Choose a code file" button to select a code file from your computer (supports C++, Python, JavaScript, Java, and more)
3. Click "Analyze" to upload and analyze the file
4. View the analysis results in the ChatGPT-like interface
5. Access your analysis history from the sidebar

### Docker Deployment

You can also run the web interface using Docker:

```bash
# Build the Docker image
docker build -t code-analyzer-web .

# Run the container
docker run -p 5000:5000 code-analyzer-web
```

Then access the web interface at http://localhost:5000.

## Troubleshooting

### Installation Issues

#### Wheel Building Errors

If you encounter errors during installation related to building wheels, such as:

```
error: subprocess-exited-with-error
× Getting requirements to build wheel did not run successfully.
```

This usually means that some packages require compilation but the necessary build tools are not available. The enhanced install_dependencies.py script will automatically try multiple installation methods:

1. First with the `--no-build-isolation` and `--prefer-binary` flags
2. Then with just the `--prefer-binary` flag
3. Then installing packages one by one with multiple fallback options:
   - First with `--prefer-binary`
   - Then with `--no-deps` and `--prefer-binary`
   - Special handling for problematic packages like modelscope and sentencepiece
   - Finally with `--only-binary=:all:` to force using pre-built wheels

If you still encounter issues, try these solutions:

1. **Use the enhanced requirements.txt**: The requirements.txt in this repository has been enhanced to use pre-built wheels where possible:
   - Added `--index-url` option to use PyPI as the primary package source
   - Added `--extra-index-url` options for PyTorch's wheel repository
   - Added `--find-links` and `--extra-index-url` options for modelscope
   - Added multiple PyTorch wheel sources for torch, sentencepiece, and numpy
   - Specified exact versions for all packages to ensure compatibility

2. **Install Visual C++ Build Tools** (Windows): Some packages require a C++ compiler. On Windows, you can install the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

3. **Install build-essential** (Linux/Mac): On Linux/Mac, install the necessary build tools:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential

   # Mac
   xcode-select --install
   ```

4. **Use the test_imports.py script**: After installation, run the test script to verify all packages can be imported:
   ```bash
   python test_imports.py
   ```

The install_dependencies.py script will automatically run this test after successful installation.

#### FileNotFoundError During Wheel Building

If you encounter a specific error like:

```
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

during wheel building, this typically means that a required executable (like a compiler) couldn't be found. Our enhanced installation process specifically addresses this issue by:

1. Preferring pre-built binary packages over source distributions that require compilation
2. Using multiple package sources to find compatible pre-built wheels
3. Providing special handling for known problematic packages
4. Using the `--only-binary=:all:` flag as a last resort to force using pre-built wheels

If you still encounter this error, make sure you have the necessary build tools installed as described above.

#### ModuleNotFoundError

If you see errors like `ModuleNotFoundError: No module named 'flask'`, it means some dependencies are missing. Use the install_dependencies.py script:

```bash
python install_dependencies.py
```

#### Authentication Errors

If you encounter errors related to authentication when installing packages, such as:

```
User for huggingface.co: ERROR: Exception:
...
EOFError: EOF when reading a line
```

This happens when pip tries to prompt for authentication credentials in a non-interactive environment. Our enhanced installation process addresses this issue by:

1. Using the `--no-input` flag with pip to prevent it from prompting for authentication
2. Using alternative package sources that don't require authentication
3. Configuring requirements.txt to use public repositories instead of authenticated ones

If you still encounter authentication issues, you can:

1. **Use the enhanced requirements.txt**: We've updated the requirements.txt file to use PyPI and PyTorch's public repositories instead of repositories that require authentication.

2. **Add the `--no-input` flag manually**: If you're installing packages manually, add the `--no-input` flag to your pip command:
   ```bash
   pip install --no-input -r requirements.txt
   ```

3. **Set up authentication beforehand**: If you need to use repositories that require authentication, set up your credentials before running the installation:
   ```bash
   pip config set global.index-url https://pypi.org/simple
   ```

### Runtime Issues

#### Syntax Checking Errors

If you encounter errors related to syntax checking, ensure the appropriate tools are installed for the languages you want to analyze:

- **C++**: 
  - **Windows**: Install MinGW or MSYS2 to get g++
  - **Linux**: `sudo apt-get install g++`
  - **Mac**: Install Xcode command line tools

- **Python**:
  - Install pylint: `pip install pylint`

- **JavaScript**:
  - Install Node.js and ESLint: `npm install -g eslint`

- **Java**:
  - Install JDK which includes javac

- **C#**:
  - Install .NET SDK which includes the C# compiler

## License

MIT
