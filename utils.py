from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "directory to store the downloaded pre-trained models"},
    )


@dataclass
class DataArguments:
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "train data file path"}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "eval data file path"},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "optimizer"})
    output_dir: Optional[str] = field(default="./output/")
    model_max_length: Optional[int] = field(
        default=1024, metadata={"help": "max length of the model. Right padded."}
    )


def process_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args
