from typing import Dict, Iterator, List, Operator

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, default_data_collator

import utils
from soho_hf.modeling_soho import SohoForCausalLM


def train():
    model_args, data_args, training_args = utils.process_args()
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model = SohoForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    # TODO: Set up train dataset
    train_dataset = None

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
