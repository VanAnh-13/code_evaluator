"""
CLI entry point for fine-tuning
"""

import argparse
import logging
import sys

from code_evaluator.finetune.trainer import finetune_model, create_sample_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the fine-tuning from command line"""
    parser = argparse.ArgumentParser(description="Fine-tune the Qwen model for code analysis")
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, help="Path to the dataset file (JSON)")
    parser.add_argument("--create_sample", action="store_true", help="Create a sample dataset")
    parser.add_argument("--sample_path", type=str, default="sample_dataset.json", help="Path to save the sample dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to create")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B-Chat", help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="fine-tuned-model", help="Directory to save the fine-tuned model")
    
    # Training arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
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
