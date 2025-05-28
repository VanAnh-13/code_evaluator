// Function to calculate factorial
function factorial(n) {
    // Missing check for negative numbers
    if (n === 0) {
        return 1;
    }
    return n * factorial(n - 1);
}

// Inefficient string concatenation
function buildString(count) {
    let result = '';
    for (let i = 0; i < count; i++) {
        result = result + 'a'; // Inefficient, should use array and join
    }
    return result;
}

// Potential memory leak with event listeners
function setupButton() {
    const button = document.getElementById('myButton');
    
    // Event listener is added but never removed
    button.addEventListener('click', function() {
        console.log('Button clicked!');
    });
}

// Security vulnerability - eval usage
function processUserInput(input) {
    // Never use eval with user input - security risk
    return eval(input);
}

// Main function with undefined variable
function main() {
    // Undefined variable
    console.log(undefinedVar);
    
    // Potential infinite recursion
    console.log(factorial(100)); // Will cause stack overflow
    
    // Inefficient string building
    const longString = buildString(10000);
    
    // Security risk
    processUserInput("alert('XSS attack')");
}

// Call main function
main();