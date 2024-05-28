from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences

class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy
    ) -> None:
        super().__init__()

        self.prompt = []
        self.label = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            self.prompt.append(data["prompt"])
            self.label.append(data["label"])

    def __len__(self):
        length = len(self.prompt)
        return length

    def __getitem__(self, idx):
        prompt, label = self.prompt[idx], self.label[idx]

        prompt_token = self.tokenizer(
            # prompt + " " + self.tokenizer.eos_token, # 因为预处理的时候已经添加过了
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        prompt_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        prompt_token["attention_mask"][0][-1] = True

        return (
            prompt_token["input_ids"],
            prompt_token["attention_mask"],
            label
        )

    def collate_fn(self, item_list):
        prompt_ids = []
        prompt_masks = []
        label = []

        for item_prompt_ids, item_prompt_mask, item_label in item_list:
            prompt_ids.append(item_prompt_ids)
            prompt_masks.append(item_prompt_mask)
            label.append(item_label)
            
        prompt_ids = zero_pad_sequences(prompt_ids, value=self.tokenizer.pad_token_id)
        prompt_masks = zero_pad_sequences(prompt_masks)

        return prompt_ids, prompt_masks, label
