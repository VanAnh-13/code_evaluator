"""
Fine-tuning script for Code Analyzer model
This script allows fine-tuning the Qwen model for code analysis tasks

NOTE: This is a backward-compatible wrapper. The main code is now in code_evaluator/finetune/

Usage:
    python finetune.py --data_path <path> [options]
    
    Or use the new unified CLI:
    python -m code_evaluator.main finetune --data <path> [options]
"""

import os
import sys

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from new package structure
from code_evaluator.finetune.trainer import finetune_model
from code_evaluator.finetune.dataset import CodeAnalysisDataset
from code_evaluator.finetune.prompt_loader import load_prompt

# Export commonly used items for backward compatibility
__all__ = [
    'finetune_model',
    'CodeAnalysisDataset',
    'create_sample_dataset',
    'load_prompt',
]


def create_sample_dataset(output_path: str, num_samples: int = 10):
    """
    Create a sample dataset for testing fine-tuning
    
    Args:
        output_path: Path to save the sample dataset
        num_samples: Number of samples to create
    """
    import json
    
    samples = []
    
    # Sample C++ code snippets with issues
    cpp_samples = [
        {
            "code": "#include <iostream>\nint main() {\n    int* ptr = new int[10];\n    // Missing delete[]\n    return 0;\n}",
            "issues": ["Memory leak: allocated memory is not freed"]
        },
        {
            "code": "#include <cstring>\nvoid copy(char* dest, const char* src) {\n    strcpy(dest, src);\n}",
            "issues": ["Buffer overflow risk: no bounds checking in strcpy"]
        },
        {
            "code": "int divide(int a, int b) {\n    return a / b;\n}",
            "issues": ["Division by zero: no check for b == 0"]
        }
    ]
    
    # Sample Python code snippets with issues
    python_samples = [
        {
            "code": "def read_file(path):\n    f = open(path)\n    data = f.read()\n    return data",
            "issues": ["Resource leak: file is not closed"]
        },
        {
            "code": "import pickle\ndef load_data(data):\n    return pickle.loads(data)",
            "issues": ["Security risk: unpickling untrusted data"]
        }
    ]
    
    all_samples = cpp_samples + python_samples
    
    for i in range(num_samples):
        sample = all_samples[i % len(all_samples)]
        samples.append({
            "instruction": "Analyze this code for bugs, security issues, and code quality problems.",
            "input": sample["code"],
            "output": "**Issues Found:**\n" + "\n".join(f"- {issue}" for issue in sample["issues"])
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    return output_path


def main():
    """Main function to run fine-tuning from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model for code analysis")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B-Chat",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training data (JSON format)")
    parser.add_argument("--output_dir", type=str, default="fine-tuned-model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--create_sample", type=str, default=None,
                        help="Create a sample dataset at the specified path")
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        create_sample_dataset(args.create_sample)
        print(f"Sample dataset created at: {args.create_sample}")
        return
    
    # Run fine-tuning
    finetune_model(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        fp16=args.fp16,
        load_in_8bit=args.load_in_8bit
    )


if __name__ == "__main__":
    main()
