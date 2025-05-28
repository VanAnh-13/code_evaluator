#!/usr/bin/env python3

from cpp_code_analyzer import check_syntax

# Test the C++ syntax checking function
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

print("Testing C++ syntax checking...")
print(f"Code to test:\n{test_code}\n")

errors = check_syntax(test_code)
print(f"Found {len(errors)} syntax errors:")
for error in errors:
    print(f"  Line {error['line']}: {error['description']}")
    print(f"    Recommendation: {error['recommendation']}")
    print()
