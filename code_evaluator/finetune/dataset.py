"""
Dataset classes for fine-tuning the Code Analyzer model
"""

import json
import logging
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from code_evaluator.finetune.prompt_loader import create_training_prompt

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
                
            # Create prompt using external templates
            language = item.get('language', 'unknown')
            prompt = create_training_prompt(code, language, analysis)
            
            self.examples.append(prompt)
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
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
    Prepare a dataset for fine-tuning using HuggingFace Dataset format
    
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
            
        # Create prompt using external templates
        language = item.get('language', 'unknown')
        prompt = create_training_prompt(code, language, analysis)
        texts.append(prompt)
    
    logger.info(f"Prepared {len(texts)} training examples")
    
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
