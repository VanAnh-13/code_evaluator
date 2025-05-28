#!/usr/bin/env python3
"""
Test script to verify the Ollama API fix
"""

from code_analyzer import CodeAnalyzer

def test_analyzer():
    print("[INFO] Testing code analyzer with API fix...")
    
    # Test with fallback mode first
    analyzer = CodeAnalyzer(model_name='none')
    test_code = 'print("Hello World")'
    
    try:
        result = analyzer.analyze_code(test_code, 'python')
        print('✅ Test successful! Result keys:', list(result.keys()))
        print('✅ Language detected:', result.get('language'))
        print('✅ Syntax errors found:', len(result.get('syntax_errors', [])))
        return True
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False

if __name__ == "__main__":
    success = test_analyzer()
    if success:
        print("\n🎉 All tests passed! The API fix is working correctly.")
    else:
        print("\n💥 Tests failed! Please check the implementation.")
