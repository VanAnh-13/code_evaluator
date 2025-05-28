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