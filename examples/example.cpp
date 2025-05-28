// example.cpp - File intentionally filled with logic and syntax errors for demonstration

#include <iostream> // Missing some includes
#include <vector>
#include <string>
using namespace std // ERROR: Missing semicolon

// ERROR: Class definition missing semicolon at end
class Student {
public:
    string name;
    int age;
    Student(string n, int a) : name(n), age(a) // ERROR: Missing braces
    void print() {
        cout << "Name: " << name << ", Age: " << age << endl // ERROR: Missing semicolon
    }
} // ERROR: Missing semicolon

// ERROR: Function with no return type
main() {
    vector<Student> students;
    students.push_back(Student("Alice", 20));
    students.push_back(Student("Bob", 21));
    students.push_back(Student("Charlie", 19));
    for (int i = 0; i <= students.size(); i++) { // ERROR: Out of bounds (should be <)
        students[i].print();
    }
    return 0;
}

// ERROR: Unused variable
int unusedVar = "string"; // ERROR: Type mismatch

// ERROR: Function never called
void doNothing(int x) {
    x = x / 0; // ERROR: Division by zero
}

// ERROR: Infinite recursion
void infinite() {
    infinite();
}

// ERROR: Function with no parameters but called with arguments
void printHello() {
    cout << "Hello" << endl;
}
void callPrintHello() {
    printHello(123); // ERROR: Too many arguments
}

// ERROR: Using undeclared variable
void useUndeclared() {
    cout << undeclaredVar << endl;
}

// ERROR: Array out of bounds
void arrayError() {
    int arr[5];
    arr[10] = 100; // ERROR: Out of bounds
}

// ERROR: Mismatched brackets
void bracketError() {
    if (true) {
        cout << "Missing closing bracket";
    // ERROR: Missing closing bracket
}

// ERROR: Using wrong type in condition
void typeError() {
    int x = 5;
    if ("string") { // ERROR: Condition should be bool/int
        cout << "Wrong type in if" << endl;
    }
}

// ERROR: Assignment in if condition
void assignmentError() {
    int x = 0;
    if (x = 5) { // ERROR: Should be ==
        cout << "Assignment in if" << endl;
    }
}

// ERROR: Function declared but not defined
void notDefined(int x);
