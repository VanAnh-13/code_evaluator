#!/usr/bin/env python3

import subprocess
import tempfile
import os
import re

def debug_cpp_syntax_check(code: str):
    """Debug version of the C++ syntax checking function"""
    syntax_errors = []

    print("Creating temporary file...")
    # Create a temporary file to store the code
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))
        print(f"Temporary file created: {temp_file_path}")

    try:
        print("Running g++ syntax check...")
        # Run g++ with -fsyntax-only to check syntax without generating output
        result = subprocess.run(
            ['g++', '-fsyntax-only', '-Wall', '-Wextra', temp_file_path],
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )

        print(f"Return code: {result.returncode}")
        print(f"Stderr output:\n{result.stderr}")

        # Parse compiler errors
        if result.returncode != 0:
            print("Parsing errors...")
            error_pattern = re.compile(r'([^:]+):(\d+):(\d+):\s+(warning|error):\s+(.+)')
            for line in result.stderr.splitlines():
                print(f"Processing line: {line}")
                match = error_pattern.match(line)
                if match:
                    file_path, line_num, column, error_type, message = match.groups()
                    print(f"  Matched: file={file_path}, line={line_num}, col={column}, type={error_type}, msg={message}")
                    severity = "high" if error_type == "error" else "medium"
                    syntax_errors.append({
                        "line": int(line_num),
                        "column": int(column),
                        "severity": severity,
                        "description": f"Syntax {error_type}: {message}",
                        "recommendation": f"Fix the {error_type} on line {line_num}"
                    })
                else:
                    print(f"  No match for: {line}")
        else:
            print("No errors found by g++")
            
    except subprocess.TimeoutExpired:
        print("Timeout expired")
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "Syntax check timed out",
            "recommendation": "The code might be too complex or contain infinite loops"
        })
    except FileNotFoundError:
        print("g++ not found")
        syntax_errors.append({
            "line": 0,
            "severity": "info",
            "description": "G++ compiler not found",
            "recommendation": "Install g++ to enable syntax checking"
        })
    finally:
        print("Cleaning up...")
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Removed temporary file: {temp_file_path}")

    return syntax_errors

# Test the debug function
test_code = '''#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Student {
public:
    string name;
    int age;
    Student(string n, int a) : name(n), age(a)
    void print() {
        cout << "Name: " << name << ", Age: " << age << endl
    }
}'''

print("Testing debug C++ syntax checking...")
print(f"Code to test:\n{test_code}\n")

errors = debug_cpp_syntax_check(test_code)
print(f"\nFound {len(errors)} syntax errors:")
for error in errors:
    print(f"  Line {error['line']}: {error['description']}")
    print(f"    Recommendation: {error['recommendation']}")
    print()
