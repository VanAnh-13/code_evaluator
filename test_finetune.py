"""
Test script for the fine-tuning functionality
This script creates a small sample dataset and runs a minimal fine-tuning job
"""

import os
import sys
import argparse
import logging
from finetune import create_sample_dataset, finetune_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test the fine-tuning functionality"""
    parser = argparse.ArgumentParser(description="Test the fine-tuning functionality")
    
    # Test configuration
    parser.add_argument("--model_name", type=str, default="qwen2:7b",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="test-fine-tuned-model", 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--sample_path", type=str, default="test_dataset.json", 
                        help="Path to save the sample dataset")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples to create")
    parser.add_argument("--use_lora", action="store_true", 
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--load_in_8bit", action="store_true", 
                        help="Load the model in 8-bit precision")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use fp16 precision")
    parser.add_argument("--num_train_epochs", type=int, default=1, 
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Create a small sample dataset
        logger.info("Creating a small sample dataset for testing...")
        create_sample_dataset(args.sample_path, args.num_samples)
        
        # Step 2: Run a minimal fine-tuning job
        logger.info("Running a minimal fine-tuning job...")
        finetune_model(
            model_name=args.model_name,
            data_path=args.sample_path,
            output_dir=args.output_dir,
            use_lora=args.use_lora,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_length=512,  # Smaller context length for testing
            save_steps=5,
            logging_steps=1,
            warmup_steps=2,
            fp16=args.fp16,
            load_in_8bit=args.load_in_8bit,
            seed=42
        )
        
        logger.info("Test completed successfully!")
        logger.info(f"Sample dataset saved to: {args.sample_path}")
        logger.info(f"Fine-tuned model saved to: {args.output_dir}")
        
        # Step 3: Verify the fine-tuned model exists
        if os.path.exists(args.output_dir):
            logger.info("Fine-tuned model directory exists.")
            
            # Check for key files
            files_to_check = ["config.json", "pytorch_model.bin"] if not args.use_lora else ["adapter_config.json", "adapter_model.bin"]
            
            for file in files_to_check:
                file_path = os.path.join(args.output_dir, file)
                if os.path.exists(file_path):
                    logger.info(f"Found expected file: {file}")
                else:
                    logger.warning(f"Missing expected file: {file}")
            
            logger.info("Fine-tuning test completed successfully!")
        else:
            logger.error(f"Fine-tuned model directory not found: {args.output_dir}")
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during fine-tuning test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())