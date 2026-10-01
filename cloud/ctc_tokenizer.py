import csv
import json
from pathlib import Path
from typing import Dict, List, Union


class CTCTokenizer:
    BLANK_TOKEN = "<blank>"
    BLANK_ID = 0

    def __init__(self, vocab: Union[str, Path, Dict[str, int]]):
        if isinstance(vocab, (str, Path)):
            with Path(vocab).open("r", encoding="utf-8") as handle:
                self.token_to_id = json.load(handle)
        else:
            self.token_to_id = vocab
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}
        if self.token_to_id.get(self.BLANK_TOKEN) != self.BLANK_ID:
            raise ValueError("<blank> must have ID 0")

    @classmethod
    def build_vocab_from_csv(cls, csv_path: Union[str, Path], output_path: Union[str, Path]):
        chars = set()
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                chars.update(row["label"].strip())
        chars.add(" ")
        vocab = {cls.BLANK_TOKEN: 0}
        vocab.update({char: index for index, char in enumerate(sorted(chars), start=1)})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        return vocab

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id[char] for char in text]

    def ctc_decode(self, ids: List[int]) -> str:
        output = []
        previous = None
        for token_id in ids:
            if token_id != previous and token_id != self.BLANK_ID:
                output.append(self.id_to_token.get(token_id, ""))
            previous = token_id
        return "".join(output)

    def __len__(self):
        return len(self.token_to_id)


def minimum_ctc_length(text: str) -> int:
    return len(text) + sum(text[i] == text[i + 1] for i in range(len(text) - 1))


def is_ctc_valid(input_length: int, text: str) -> bool:
    return input_length >= minimum_ctc_length(text)
