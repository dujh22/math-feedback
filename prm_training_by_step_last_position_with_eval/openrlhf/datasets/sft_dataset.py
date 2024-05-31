from typing import Callable
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences

class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        pretrain_mode=False,
    ) -> None:
        super().__init__()

        self.prompts = []
        self.prompt_ids_lens = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            question = data["question"]
            response = data["response"]
            response = response.replace("ки", "<|reserved_special_token_250|>")
            prompt = "Question: " + question + "\nAnswer: " + response
            
            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            else:
                prompt_ids_len = 0

            if not self.pretrain_mode:
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not response:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]

        input_token = self.tokenizer(
            prompt + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"prompt": prompt}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"prompt": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["prompt"].append(info["prompt"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")

        return prompt_ids_lens, input_ids, attention_masks, infos
