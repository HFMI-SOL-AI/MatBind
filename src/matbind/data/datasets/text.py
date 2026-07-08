import os

import pandas as pd
from torch.utils.data import Dataset
from transformers.models.bert import BertTokenizerFast

from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


def instantiate_bert_tokenizer(
    pretrained_model_name_or_path: str,
    do_lower_case: bool,
    tokenizer_parallelism: bool = False,
):
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = str(tokenizer_parallelism).lower()
        tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, do_lower_case=do_lower_case
        )
        return tokenizer

    except OSError as e:
        LOGGER.error(
            f"Bert tokenizer initialization failed: {e!s} You can ignore this error if you are not using a text encoder."
        )
        return None


class TextDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        partial_text: bool = False,
    ):
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")

        self.text_data = data["text"].to_list()
        LOGGER.debug("Initializing TextDataset")
        LOGGER.debug(f"Data columns: {data.columns}")
        LOGGER.debug(f"Sample text: {self.text_data[0]}")
        LOGGER.debug(f"Text Dataset Size: {len(self.text_data)}")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.partial_text = partial_text
        LOGGER.info(f"Partial text enabled: {self.partial_text}")
        LOGGER.info(f"Max tokenization length: {self.max_length}")

        self.test_tokenizer()

    def test_tokenizer(self):
        try:
            sample_text = self.text_data[0]
            sample_encoding = self.tokenizer(
                sample_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            LOGGER.debug(f"Sample tokenization successful. Output shape: {sample_encoding['input_ids'].shape}")
        except Exception as e:
            LOGGER.error(f"Tokenizer test failed: {e!s}")
            raise e

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        try:
            data = self.text_data[idx]
            if self.partial_text:
                data = data.split(".")[0] + "."  # Truncate text to max_length characters before tokenization
            encoding = self.tokenizer(
                data,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            LOGGER.error(f"Tokenization failed for index {idx}: {e!s}")
            LOGGER.error(f"Text content: {self.text_data[idx]}")
            raise e

        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
        )


if __name__ == "__main__":
    # Example usage
    data = pd.read_parquet("/p/project1/solai/datasets/materials_project/text.parquet")
    tokenizer = instantiate_bert_tokenizer("/p/project1/solai/datasets/matbert-base-cased", do_lower_case=True)
    dataset = TextDataset(data, tokenizer, max_length=50, partial_text=True)