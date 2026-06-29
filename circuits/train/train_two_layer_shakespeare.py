import os
import time
from functools import partial

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode as CN

from circuits.models.two_attn_layer import TwoLayerAttnTransformer
from circuits.train.trainer import Trainer
from circuits.train.utils import set_seed, setup_logging

# absolute paths so this runs from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = os.path.join(REPO_ROOT, 'out', 'shakespeare_2layer')

    # model -- small so it trains on MPS/CPU in minutes
    C.model = TwoLayerAttnTransformer.get_default_config()
    C.model.vocab_size = 50257 + 1  # The +1 is for the extra start token
    C.model.n_embd = 128
    C.model.n_head = 4
    C.model.pos_embd_pdrop = 0.0

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # 'auto' for cuda
    C.trainer.block_size = 128
    C.trainer.batch_size = 32
    C.trainer.micro_batch_size = 32

    C.trainer.learning_rate = 1e-3
    C.trainer.decay_lr = True
    C.trainer.warmup_iters = 100
    C.trainer.lr_decay_iters = 3000
    C.trainer.min_lr = 1e-4
    C.trainer.max_iters = 3000

    C.trainer.start_token = 50257
    return C


def batch_end_callback(trainer, writer, config):
    if trainer.iter_num % 10 == 0:
        writer.add_scalar('train_loss', trainer.loss, trainer.iter_num)
        writer.add_scalar('learning_rate', trainer.current_lr, trainer.iter_num)

    if trainer.iter_num % 250 == 0:
        val_loss = trainer.validate()
        writer.add_scalar('val_loss', val_loss, trainer.iter_num)
        tqdm.write(f"iter {trainer.iter_num} val loss: {val_loss:.4f}")

    if trainer.iter_num % 1000 == 0:
        ckpt_path = os.path.join(config.system.work_dir, f"latest_model_{trainer.iter_num}.pt")
        torch.save(trainer.model.state_dict(), ckpt_path)
        tqdm.write(f"saved checkpoint at iter {trainer.iter_num} -> {ckpt_path}")


def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    config = get_config()
    print(config)

    set_seed(config.system.seed)
    setup_logging(config)

    writer = SummaryWriter(os.path.join(config.system.work_dir, 'tensorboard', time.strftime("%Y-%m-%d_%H-%M-%S")))

    data_dir = os.path.join(REPO_ROOT, 'data', 'shakespeare')
    if not os.path.exists(os.path.join(data_dir, 'train.bin')):
        raise ValueError("data not found, please run: python -m circuits.train.shakespeare")

    config.model.block_size = config.trainer.block_size
    model = TwoLayerAttnTransformer(config.model)

    trainer = Trainer(config.trainer, model, data_dir=data_dir)

    trainer.add_callback('on_batch_end',
        partial(batch_end_callback, writer=writer, config=config)
    )

    trainer.run()


if __name__ == '__main__':
    train()
