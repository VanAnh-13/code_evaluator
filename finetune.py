"""
Fine-tuning module for the Code Analyzer
This module provides functionality to fine-tune the Qwen model on custom code analysis datasets
"""

import sys
import json
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_scheduler,
    default_data_collator
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from datasets import load_dataset, Dataset as HFDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CodeAnalysisDataset(Dataset):
    """
    Dataset for fine-tuning the model on code analysis tasks
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the dataset file (JSON)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process data
        for item in data:
            code = item.get('code', '')
            analysis = item.get('analysis', '')

            if not code or not analysis:
                continue

            # Create prompt using the same format as in the analyzer
            language = item.get('language', 'unknown')
            prompt = self._create_prompt(code, language)

            # Combine prompt and analysis
            text = f"{prompt}\n{analysis}"
            self.examples.append(text)

        logger.info(f"Loaded {len(self.examples)} examples")

    def _create_prompt(self, code: str, language: str) -> str:
        """
        Create a prompt for the model based on the language
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Formatted prompt
        """
        # Use the same prompt templates as in the CodeAnalyzer class
        if language == "cpp":
            prompt = f"""
            You are an expert C++ code analyzer. Analyze the following C++ code for:
            1. Potential bugs and logical errors
            2. Memory management issues (leaks, dangling pointers, etc.)
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            C++ CODE:
            ```cpp
            {code}
            ```

            ANALYSIS:
            """
        elif language == "python":
            prompt = f"""
            You are an expert Python code analyzer. Analyze the following Python code for:
            1. Potential bugs and logical errors
            2. Memory and resource management issues
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues (PEP 8 compliance)

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            PYTHON CODE:
            ```python
            {code}
            ```

            ANALYSIS:
            """
        elif language == "javascript":
            prompt = f"""
            You are an expert JavaScript code analyzer. Analyze the following JavaScript code for:
            1. Potential bugs and logical errors
            2. Memory leaks and resource management
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            JAVASCRIPT CODE:
            ```javascript
            {code}
            ```

            ANALYSIS:
            """
        else:
            # Default prompt for other languages
            prompt = f"""
            You are an expert code analyzer. Analyze the following code for:
            1. Potential bugs and logical errors
            2. Resource management issues
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            CODE:
            ```
            {code}
            ```

            ANALYSIS:
            """

        return prompt.strip()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize the text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encodings.items()}

        # For causal language modeling, labels are the same as input_ids
        item["labels"] = item["input_ids"].clone()

        return item


def prepare_dataset(data_path: str, tokenizer, max_length: int = 2048) -> HFDataset:
    """
    Prepare a dataset for fine-tuning
    
    Args:
        data_path: Path to the dataset file (JSON)
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace dataset
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process data
    texts = []
    for item in data:
        code = item.get('code', '')
        analysis = item.get('analysis', '')

        if not code or not analysis:
            continue

        # Create prompt using the same format as in the analyzer
        language = item.get('language', 'unknown')

        # Use the same prompt templates as in the CodeAnalyzer class
        if language == "cpp":
            prompt = f"""
            You are an expert C++ code analyzer. Analyze the following C++ code for:
            1. Potential bugs and logical errors
            2. Memory management issues (leaks, dangling pointers, etc.)
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            C++ CODE:
            ```cpp
            {code}
            ```

            ANALYSIS:
            {analysis}
            """
        elif language == "python":
            prompt = f"""
            You are an expert Python code analyzer. Analyze the following Python code for:
            1. Potential bugs and logical errors
            2. Memory and resource management issues
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues (PEP 8 compliance)

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            PYTHON CODE:
            ```python
            {code}
            ```

            ANALYSIS:
            {analysis}
            """
        elif language == "javascript":
            prompt = f"""
            You are an expert JavaScript code analyzer. Analyze the following JavaScript code for:
            1. Potential bugs and logical errors
            2. Memory leaks and resource management
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            JAVASCRIPT CODE:
            ```javascript
            {code}
            ```

            ANALYSIS:
            {analysis}
            """
        else:
            # Default prompt for other languages
            prompt = f"""
            You are an expert code analyzer. Analyze the following code for:
            1. Potential bugs and logical errors
            2. Resource management issues
            3. Security vulnerabilities
            4. Performance issues
            5. Code style and readability issues

            Provide a detailed analysis with specific line numbers and recommendations for improvement.

            CODE:
            ```
            {code}
            ```

            ANALYSIS:
            {analysis}
            """

        texts.append(prompt.strip())

    # Create HuggingFace dataset
    dataset = HFDataset.from_dict({"text": texts})

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def finetune_model(
        model_name: str,
        data_path: str,
        output_dir: str,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 2048,
        save_steps: int = 100,
        logging_steps: int = 10,
        warmup_steps: int = 100,
        fp16: bool = True,
        bf16: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        seed: int = 42
):
    """
    Fine-tune the model on a custom dataset
    
    Args:
        model_name: Name or path of the pre-trained model
        data_path: Path to the dataset file (JSON)
        output_dir: Directory to save the fine-tuned model
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_length: Maximum sequence length
        save_steps: Save checkpoint every X steps
        logging_steps: Log every X steps
        warmup_steps: Number of warmup steps
        fp16: Whether to use fp16 precision
        bf16: Whether to use bf16 precision
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model: {model_name}")

    # Determine quantization configuration
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16 if fp16 else torch.bfloat16 if bf16 else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if fp16 else torch.bfloat16 if bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Prepare model for k-bit training if using quantization
    if load_in_8bit or load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if requested
    if use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Prepare dataset
    logger.info("Preparing dataset")
    dataset = prepare_dataset(data_path, tokenizer, max_length)

    # Split dataset into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(eval_dataset)}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        save_total_limit=3,
        fp16=fp16,
        bf16=bf16,
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    logger.info("Starting training")
    trainer.train()

    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Fine-tuning completed successfully")


def create_sample_dataset(output_path: str, num_samples: int = 10):
    """
    Create a sample dataset for fine-tuning
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to create
    """
    # Sample C++ code
    cpp_code = """
    #include <iostream>
    #include <vector>
    #include <string>
    
    void process_data(std::vector<int>& data) {
        for (int i = 0; i <= data.size(); i++) {
            std::cout << data[i] << std::endl;
        }
    }
    
    int main() {
        std::vector<int> numbers = {1, 2, 3, 4, 5};
        process_data(numbers);
        
        int* ptr = new int[10];
        // Missing delete[] ptr
        
        return 0;
    }
    """

    # Sample Python code
    python_code = """
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
    """

    # Sample JavaScript code
    js_code = """
    function processData(data) {
        for (let i = 0; i <= data.length; i++) {
            console.log(data[i]);
        }
    }
    
    function main() {
        const numbers = [1, 2, 3, 4, 5];
        processData(numbers);
        
        // Potential memory leak
        const elements = document.getElementsByClassName('item');
        for (let i = 0; i < elements.length; i++) {
            elements[i].addEventListener('click', function() {
                console.log('Item clicked');
            });
        }
    }
    
    main();
    """

    # Sample analyses
    cpp_analysis = """
    I've analyzed the C++ code and found several issues:
    
    1. Line 6: Off-by-one error in the loop condition. Using <= with data.size() will cause an out-of-bounds access in the last iteration. Change to < instead of <=.
    
    2. Line 15-16: Memory leak. The code allocates memory with new int[10] but never deallocates it with delete[]. Add delete[] ptr before the return statement.
    
    3. Line 6-8: Inefficient loop. Consider using a range-based for loop for better readability and to avoid potential indexing errors:
       for (const auto& value : data) {
           std::cout << value << std::endl;
       }
    
    4. Line 5: The function takes a vector by reference but doesn't modify it. Consider using const reference (const std::vector<int>& data) to make the intent clear.
    
    5. General: The code lacks error handling and input validation, which could lead to runtime issues in a production environment.
    """

    python_analysis = """
    I've analyzed the Python code and found several issues:
    
    1. Line 9: Potential division by zero error if the input list is empty. Add a check to ensure count is not zero before division.
    
    2. Line 13: Using an undefined variable 'user_input'. Define this variable before using it or remove this line.
    
    3. Line 16-18: Inefficient list creation. Use list comprehension instead: result = [i * i for i in range(1000)]
    
    4. Line 21-23: File resource not properly closed. Use a 'with' statement to ensure the file is closed properly:
       with open("data.txt", "r") as f:
           content = f.read()
           print(content)
    
    5. Line 26: Calling calculate_average with an empty list will cause a division by zero error. Add error handling or check for empty lists.
    
    6. General: The code lacks docstrings and type hints, which would improve maintainability and help catch errors during development.
    """

    js_analysis = """
    I've analyzed the JavaScript code and found several issues:
    
    1. Line 2: Off-by-one error in the loop condition. Using <= with data.length will cause an undefined access in the last iteration. Change to < instead of <=.
    
    2. Line 12-18: Potential memory leak due to event listener binding inside a loop. This creates a closure for each element, which can lead to memory issues in large applications. Consider using event delegation instead:
       document.addEventListener('click', function(event) {
           if (event.target.classList.contains('item')) {
               console.log('Item clicked');
           }
       });
    
    3. Line 2-4: Inefficient loop. Consider using forEach or map for better readability:
       data.forEach(item => console.log(item));
    
    4. General: The code lacks error handling. For example, if 'data' is null or undefined, the function will throw an error.
    
    5. General: No input validation is performed before processing the data, which could lead to runtime errors.
    """

    # Create dataset
    dataset = []

    # Add C++ samples
    for i in range(num_samples // 3 + 1):
        dataset.append({
            "language": "cpp",
            "code": cpp_code,
            "analysis": cpp_analysis
        })

    # Add Python samples
    for i in range(num_samples // 3 + 1):
        dataset.append({
            "language": "python",
            "code": python_code,
            "analysis": python_analysis
        })

    # Add JavaScript samples
    for i in range(num_samples // 3 + 1):
        dataset.append({
            "language": "javascript",
            "code": js_code,
            "analysis": js_analysis
        })

    # Save dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Created sample dataset with {len(dataset)} examples at {output_path}")


def main():
    """Main function to run the fine-tuning from command line"""
    parser = argparse.ArgumentParser(description="Fine-tune the Qwen model for code analysis")

    # Dataset arguments
    parser.add_argument("--data_path", type=str, help="Path to the dataset file (JSON)")
    parser.add_argument("--create_sample", action="store_true", help="Create a sample dataset")
    parser.add_argument("--sample_path", type=str, default="sample_dataset.json",
                        help="Path to save the sample dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to create")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="https://f6bf-34-69-56-104.ngrok-free.app/Qwen-27B",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="fine-tuned-model",
                        help="Directory to save the fine-tuned model")

    # Training arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")

    # Precision arguments
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load the model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the model in 4-bit precision")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create sample dataset if requested
    if args.create_sample:
        create_sample_dataset(args.sample_path, args.num_samples)
        logger.info(f"Sample dataset created at {args.sample_path}")
        if not args.data_path:
            args.data_path = args.sample_path

    # Check if data path is provided
    if not args.data_path:
        parser.error("Either --data_path or --create_sample must be provided")

    # Fine-tune the model
    finetune_model(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
