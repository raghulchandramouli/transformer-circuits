# Tiny Shakespeare data prep. Mirrors Karpathy's nanoGPT shakespeare prepare.py,
# but writes the same {train,val}.bin uint16 memmap format our Trainer expects.

import os
from urllib.request import urlretrieve

import numpy as np
import tiktoken

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def save_dataset(data_dir):
    input_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(input_path):
        print('downloading tiny shakespeare...')
        urlretrieve(DATA_URL, input_path)

    with open(input_path, 'r') as f:
        data = f.read()

    # 90/10 train/val split on the raw text
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode with the gpt2 bpe tokenizer (same as openwebtext.py)
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files (uint16 since gpt2 max token id 50256 < 2**16)
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(data_dir, 'train.bin'))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(data_dir, 'val.bin'))
    print(f"wrote train.bin and val.bin to {data_dir}")


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, '..', '..', 'data', 'shakespeare')
    os.makedirs(data_dir, exist_ok=True)
    save_dataset(data_dir)
