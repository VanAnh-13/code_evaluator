#include <iostream>
using namespace std;

// Hàm tính tổng hai số nguyên
int addNumbers(int a, b) { // Lỗi: Thiếu kiểu dữ liệu cho tham số `b`
    return a + b;
}

int main() {
    int x = 10;
    int y = "20"; // Lỗi: Gán chuỗi vào biến kiểu số nguyên

    cout << "Tổng của x và y là: " << addNumbers(x, y) << endl;

    char name[5]; // Mảng chỉ có thể chứa 5 ký tự
    cout << "Nhập tên của bạn: ";
    cin >> name; // Lỗi tiềm ẩn: Nếu người dùng nhập quá 5 ký tự, sẽ gây tràn bộ nhớ

    if (x = 20) { // Lỗi logic: Sử dụng toán tử gán (=) thay vì so sánh (==)
        cout << "x bằng 20" << endl;
    } else {
        cout << "x không bằng 20" << endl;
    }

    for (int i = 0; i <= 10; i++) { // Lỗi logic: Điều kiện nên là `i < 10` nếu muốn lặp 10 lần
        cout << "Giá trị của i: " << i << endl;
    }

    int* ptr = nullptr;
    cout << "Giá trị tại con trỏ: " << *ptr << endl; // Lỗi thời gian chạy: Truy cập vào con trỏ null

    return 0;
}