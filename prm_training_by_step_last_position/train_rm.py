# import sys
# import os
# # 获取当前脚本的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 如果当前目录不在sys.path中，就将其添加到sys.path中
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# # # 导入其他模块
# sys.path.append('/root/miniconda3/envs/openrlhf/lib/python3.10/site-packages')
# os.environ['WANDB__EXECUTABLE'] = '/root/miniconda3/envs/openrlhf/bin/python'

# import subprocess

import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import RewardDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer import RewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        train_data_dir=args.train_data_dir,
        test_data_dir=args.test_data_dir,
        probabilities=args.dataset_probs,
        strategy=strategy,
        seed=args.seed,
        stopping_strategy="all_exhausted",
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = RewardDataset(train_data, tokenizer, args.max_len, strategy)
    eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len, strategy)

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, 
        args.micro_train_batch_size, 
        True, 
        False, 
        eval_dataset.collate_fn
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
    )
    
    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default="/workspace/dujh22/models/llama3_8B_rw_debug")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--pretrain", type=str, default="/workspace/dujh22/models/llama-3-8B")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    parser.add_argument("--train_data_dir", type=str, default="/workspace/dujh22/math_feedback/prm_training/raw_data/math_shepherd2/train.jsonl")
    parser.add_argument("--test_data_dir", type=str, default="/workspace/dujh22/math_feedback/prm_training/raw_data/math_shepherd2/test.jsonl")
    parser.add_argument("--dataset_probs", type=str, default="1", help="sampling probs for datasets")
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--use_wandb", type=str, default="True")



    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default=None)
    # wandb pamameters
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)
