"""
Training utilities for fine-tuning the Code Analyzer model
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from code_evaluator.finetune.dataset import prepare_dataset

logger = logging.getLogger(__name__)


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
    
    cpp_analysis = """
    I've analyzed the C++ code and found several issues:
    
    1. Line 6: Off-by-one error in the loop condition. Using <= with data.size() will cause an out-of-bounds access in the last iteration. Change to < instead of <=.
    
    2. Line 15-16: Memory leak. The code allocates memory with new int[10] but never deallocates it with delete[]. Add delete[] ptr before the return statement.
    
    3. Line 6-8: Inefficient loop. Consider using a range-based for loop for better readability and to avoid potential indexing errors.
    
    4. Line 5: The function takes a vector by reference but doesn't modify it. Consider using const reference (const std::vector<int>& data) to make the intent clear.
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
    
    if __name__ == "__main__":
        main()
    """
    
    python_analysis = """
    I've analyzed the Python code and found several issues:
    
    1. Line 9: Potential division by zero error if the input list is empty. Add a check to ensure count is not zero before division.
    
    2. Line 13: Using an undefined variable 'user_input'. Define this variable before using it or remove this line.
    
    3. Line 16-18: Inefficient list creation. Use list comprehension instead: result = [i * i for i in range(1000)]
    
    4. Line 21-23: File resource not properly closed. Use a 'with' statement to ensure the file is closed properly.
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
    
    js_analysis = """
    I've analyzed the JavaScript code and found several issues:
    
    1. Line 2: Off-by-one error in the loop condition. Using <= with data.length will cause an undefined access in the last iteration. Change to < instead of <=.
    
    2. Line 12-18: Potential memory leak due to event listener binding inside a loop. Consider using event delegation instead.
    
    3. Line 2-4: Inefficient loop. Consider using forEach or map for better readability.
    
    4. General: The code lacks error handling. If 'data' is null or undefined, the function will throw an error.
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
    
    # Ensure output directory exists
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Created sample dataset with {len(dataset)} examples at {output_path}")
