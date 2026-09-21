import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):

    def __init__(self, txt_path, max_seq_len,tokenizer,label2id):
        self.txt_path = txt_path
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len
        self.samples = []
        self._load_data()

    # 读取数据封装成一个函数当成员函数，init调用
    def _load_data(self):
        tokens = []
        labels = []
        with open(
                self.txt_path,
                "r",
                encoding="utf-8"
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        self.samples.append({
                            "tokens": tokens,
                            "labels": labels
                        })
                        tokens = []
                        labels = []
                    continue
                parts = line.split()

                if len(parts) < 2:
                    continue

                token = parts[0]
                label = parts[-1]

                tokens.append(token)
                labels.append(label)
        # 考虑最后一个句子后没有空行
        if tokens:
            self.samples.append({
                "tokens": tokens,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        labels = sample["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(
            batch_index=0
        )
        aligned_labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(
                    self.label2id[
                        labels[word_id]
                    ]
                )
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id
        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),
            "attention_mask":
                encoding["attention_mask"].squeeze(0),
            "token_type_ids":
                encoding["token_type_ids"].squeeze(0)
                if "token_type_ids" in encoding
                else torch.zeros_like(
                    encoding["input_ids"].squeeze(0)
                ),

            "labels":
                torch.tensor(
                    aligned_labels,
                    dtype=torch.long
                )
        }

def build_label_map(txt_path):

    raw_labels = set()
    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = parts[-1]
            raw_labels.add(label)
    raw_labels.discard("O")
    raw_labels = sorted(raw_labels)
    labels = ["O"] + raw_labels

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    return label2id, id2label