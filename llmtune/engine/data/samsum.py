import torch
from typing import Dict, Any
from datasets import load_dataset
from llmtune.engine.data.abstract import AbstractTrainData

DEFAULT_HF_PATH = "samsum"

class TrainSAMsum(AbstractTrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            # if (
            #     result["input_ids"][-1] != self.tokenizer.eos_token_id
            #     and len(result["input_ids"]) < self.cutoff_len
            # ):
            #     result["input_ids"].append(self.tokenizer.eos_token_id)
            #     result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        if self.dataset:
            data = load_dataset("json", data_files=self.dataset)
        else:
            data = load_dataset(DEFAULT_HF_PATH)


        self.train_data = data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        self.val_data = data["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))


    # Auxiliary methods
    def generate_prompt(self, data_point, **kwargs):
        return make_prompt(
            data_point["dialogue"],
            data_point["summary"],
        )


    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)

def make_prompt(dialogue, summary=""):
    return "{0}\n{1}\n\n{2}\n{3}".format(
        "### Dialogue:",
        dialogue,
        "### Summary:",
        summary,
    )

def make_output(raw_output):
    return raw_output.split("### Summary:")[1].strip()