import math
from abc import ABC

import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss, SwitchBalancingLoss

from torch import distributed as dist

import numpy as np

class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            acc_sum = 0
            cor_sum = 0
            acc_sum2 = 0
            cor_sum2 = 0
            id = 0
            for prompt_ids, prompt_masks, label in self.train_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_masks = prompt_masks.squeeze(1).to(torch.cuda.current_device())
                label_tensor = torch.tensor(label, dtype=torch.float, device=torch.cuda.current_device()).squeeze()

                prompt_rewards, output = self.model(prompt_ids, attention_mask=prompt_masks, return_output=True)                                
                # 将列表转换为PyTorch张量
                predictions_tensor = prompt_rewards
                targets_tensor = label_tensor

                # 避免 log(0) 的情况，将预测值限制在一个很小的范围内
                # epsilon = 1e-12
                # predictions_tensor = torch.clamp(predictions_tensor, epsilon, 1. - epsilon)
                # 计算交叉熵损失
                loss = targets_tensor * torch.log(predictions_tensor) + (1 - targets_tensor) * torch.log(1 - predictions_tensor)
                loss = -torch.mean(loss)

                acc = (predictions_tensor.round() == label_tensor).float().mean().item()

                # # 确保张量数据类型匹配
                # if predictions_tensor.dtype != torch.float32:
                #     predictions_tensor = predictions_tensor.float()  # 将预测结果转换为Float类型
                # if targets_tensor.dtype != torch.float32:
                #     targets_tensor = targets_tensor.float()  # 确保目标张量也是Float类型
                # # 检查并调整预测张量和目标张量的形状以匹配
                # predictions_tensor = predictions_tensor.view(-1)
                # targets_tensor = targets_tensor.view(-1) 

                # criterion = nn.BCELoss()
                # loss = criterion(predictions_tensor, targets_tensor)

                # acc = (predictions_tensor.round() == targets_tensor).float().mean().item()
                
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                if not np.isnan(acc):
                    acc_mean = acc_mean * 0.9 + 0.1 * acc
                if np.isnan(acc_mean):
                    acc_mean = 0

                loss_value = loss.item()
                if not np.isnan(loss_value):
                    loss_mean = loss_mean * 0.9 + 0.1 * loss_value
                if np.isnan(loss_mean):
                    loss_mean = 0

                if targets_tensor.dim() > 0:
                    if id % args.train_batch_size == 0:
                        acc_sum2 = 0
                        cor_sum2 = 0
                    acc_sum = acc_sum + targets_tensor.size(0)
                    cor_sum = cor_sum + (predictions_tensor.round() == targets_tensor).sum().item()
                    acc_sum2 = acc_sum2 + targets_tensor.size(0)
                    cor_sum2 = cor_sum2 + (predictions_tensor.round() == targets_tensor).sum().item()
                acc_all = cor_sum / acc_sum
                acc_all2 = cor_sum2 / acc_sum2

                # optional rm info
                logs_dict = {
                    "acc": acc,
                    "loss": loss.item(),
                    "prompt_rewards": prompt_rewards.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                    "acc_all": acc_all,
                    "acc_all2": acc_all2,
                }

                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
                id += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        # if global_step % args.eval_steps == 0:
        #     self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            for prompt_ids, prompt_masks, label in eval_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_masks = prompt_masks.squeeze(1).to(torch.cuda.current_device())
                label_tensor = torch.tensor(label, dtype=torch.float, device=torch.cuda.current_device()).squeeze()

                prompt_rewards, output = self.model(prompt_ids, attention_mask=prompt_masks, return_output=True)                                
                # 将列表转换为PyTorch张量
                predictions_tensor = prompt_rewards
                targets_tensor = label_tensor
                # 避免 log(0) 的情况，将预测值限制在一个很小的范围内
                epsilon = 1e-12
                predictions_tensor = torch.clamp(predictions_tensor, epsilon, 1. - epsilon)
                # 计算交叉熵损失
                loss = targets_tensor * torch.log(predictions_tensor) + (1 - targets_tensor) * torch.log(1 - predictions_tensor)
                loss = -torch.sum(loss)

                rewards += [prompt_rewards.flatten()]
                acc += (predictions_tensor.round() == label_tensor).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()
            
            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict) # 会卡死，这是个没有解决的bug，多进程没有同步
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

