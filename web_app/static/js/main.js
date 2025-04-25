/**
 * Main JavaScript file for the C++ Code Analyzer web application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.display = 'none';
        }, 5000);
    });

    // Scroll to bottom of chat messages on page load
    const chatMessages = document.querySelector('.chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Add syntax highlighting to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    if (codeBlocks.length > 0 && typeof hljs !== 'undefined') {
        codeBlocks.forEach(block => {
            hljs.highlightElement(block);
        });
    }

    // Make issue items collapsible
    const issueItems = document.querySelectorAll('.issue-item');
    issueItems.forEach(function(item) {
        const header = item.querySelector('.issue-header');
        if (header) {
            header.addEventListener('click', function() {
                item.classList.toggle('collapsed');
            });
        }
    });

    // Add event listener for file input change
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0] ? this.files[0].name : 'Choose a C++ file';
            const fileNameElement = document.getElementById('file-name');
            if (fileNameElement) {
                fileNameElement.textContent = fileName;
            }
            
            // Enable submit button if file is selected
            const submitButton = document.querySelector('.submit-button');
            if (submitButton) {
                submitButton.disabled = !this.files[0];
            }
        });
    }

    // Disable submit button initially if no file is selected
    const fileInputInitial = document.getElementById('file-input');
    const submitButtonInitial = document.querySelector('.submit-button');
    if (fileInputInitial && submitButtonInitial) {
        submitButtonInitial.disabled = !fileInputInitial.files[0];
    }
});