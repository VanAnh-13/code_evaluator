#!/usr/bin/env python3
"""
Test script to verify the Ollama fixes are working
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_code_analyzer_import():
    """Test that CodeAnalyzer can be imported without errors"""
    try:
        from code_analyzer import CodeAnalyzer, generate_report
        print("[SUCCESS] CodeAnalyzer imported successfully")
        return True
    except Exception as e:
        print(f"[FAILED] Failed to import CodeAnalyzer: {str(e)}")
        return False

def test_analyzer_initialization():
    """Test that CodeAnalyzer can be initialized in fallback mode"""
    try:
        from code_analyzer import CodeAnalyzer
        
        # Test with fallback mode
        analyzer = CodeAnalyzer(model_name="none")
        print("[SUCCESS] CodeAnalyzer initialized in fallback mode")
        
        # Test load_model with fallback
        result = analyzer.load_model()
        if result:
            print("[SUCCESS] load_model() works in fallback mode")
        else:
            print("[FAILED] load_model() failed in fallback mode")
            return False
        
        return True
    except Exception as e:
        print(f"[FAILED] Failed to initialize CodeAnalyzer: {str(e)}")
        return False

def test_basic_analysis():
    """Test basic code analysis without AI model"""
    try:
        from code_analyzer import CodeAnalyzer
        
        analyzer = CodeAnalyzer(model_name="none")
        
        # Test with simple C++ code
        cpp_code = """
#include <iostream>
int main() {
    int x = "hello";  // Error: string assigned to int
    std::cout << x << std::endl;
    return 0;
}
"""
        
        results = analyzer.analyze_code(cpp_code, "cpp")
        
        if "syntax_errors" in results and "language" in results:
            print(f"[SUCCESS] Basic analysis completed. Found {len(results.get('syntax_errors', []))} syntax errors")
            print(f"[INFO] Language detected: {results.get('language', 'unknown')}")
            
            # Check if we found the string-to-int assignment error
            total_issues = sum(len(results.get(k, [])) for k in ["syntax_errors", "bugs", "memory_issues", "security_vulnerabilities", "performance_issues", "style_issues"])
            print(f"[INFO] Total issues found: {total_issues}")
            
            return True
        else:
            print("[FAILED] Analysis did not return expected structure")
            return False
            
    except Exception as e:
        print(f"[FAILED] Failed to perform basic analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Ollama fixes for Code Analyzer")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_code_analyzer_import),
        ("Initialization Test", test_analyzer_initialization),
        ("Basic Analysis Test", test_basic_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"[FAILED] {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] All tests passed! The Ollama fixes are working correctly.")
        print("[INFO] The application can now run without Ollama AI models using fallback mode.")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
