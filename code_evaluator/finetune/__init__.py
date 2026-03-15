"""
Fine-tuning module for Code Evaluator
"""

def __getattr__(name):
    """Lazy import for fine-tuning classes that require torch"""
    if name == "CodeAnalysisDataset":
        from code_evaluator.finetune.dataset import CodeAnalysisDataset
        return CodeAnalysisDataset
    elif name == "prepare_dataset":
        from code_evaluator.finetune.dataset import prepare_dataset
        return prepare_dataset
    elif name == "finetune_model":
        from code_evaluator.finetune.trainer import finetune_model
        return finetune_model
    elif name == "create_sample_dataset":
        from code_evaluator.finetune.trainer import create_sample_dataset
        return create_sample_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CodeAnalysisDataset",
    "prepare_dataset",
    "finetune_model",
    "create_sample_dataset",
]
