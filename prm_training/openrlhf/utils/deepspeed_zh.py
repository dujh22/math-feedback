# 导入所需的Python模块和函数
import os
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

# 导入相关的深度学习库
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

# 导入自定义的模型和工具
from openrlhf.models import Actor
from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)

# 定义类型别名，用于清晰表达函数返回值和参数类型
ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

# 定义一个基于DeepSpeed的训练策略类
class DeepspeedStrategy(ABC):
    """
    用于加速器训练的策略。
    """

    # 初始化函数
    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        zero_stage=2,
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        # 设置训练参数和状态
        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", "fp32")
        self.disable_trace_cache = getattr(args, "disable_trace_cache", False)

        # 用于记录训练步骤
        self.is_rlhf = False
        self.time_steps = defaultdict(int)

    # 设置随机种子以确保可复现性
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置分布式训练环境
    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)

        # 初始化分布式训练后端
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    # 创建优化器
    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    # 定义反向传播过程
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    # 优化器步骤，进行模型更新
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.step()

    # 设置数据加载器
    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
    ):
        if sampler is None:
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
            )

        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    # 移除模型包装，用于获取实际的模型
    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    # 准备模型和优化器，用于训练或评估
    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    # 初始化训练模型配置
    def _ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_train_config(is_actor)

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    # 获取训练模型的DeepSpeed配置
    def get_ds_train_config(self, is_actor):
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            disable_trace_cache=self.disable_trace_cache,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        train_batch_size = self.train_batch_size
        if self.is_rlhf and is_actor and self.args.pretrain_data is not None:
            train_batch_size *= 2
        ds_config["train_batch_size"] = train_batch_size

        return ds_config

    # 初始化评估模型配置
    def _ds_init_eval_model(self, model):
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_eval_config(offload=getattr(model, "_offload", False))

        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor else model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model

    # 获取评估模型的DeepSpeed配置
    def get_ds_eval_config(self, offload=False):
        ds_config = get_eval_ds_config(offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16)
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size

        return ds_config

    # 更新模型的指数移动平均权重
    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: 使用预筛选以提高效率
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    # 加载模型
    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    # 保存模型
    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        model_to_save = self._unwrap_model(model)

        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv

        if self.is_rank_0():
            state_dict = model_to_save.state_dict()

            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())
            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)

            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(output_dir)

            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(train_from_model_path, filename), os.path.join(output_dir, filename))

    # 进行数据归约操作，支持mean、max和sum三种操作
    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    # 收集所有GPU中的数据
    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    # 打印信息，仅在rank 0上执行
    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    # 检查当前是否为rank 0
    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    # 获取当前的rank编号
    def get_rank(self) -> int:
        return dist.get_rank()

    # 保存检查点
    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        if self.is_rank_0():
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                subdirs = [
                    (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                    for d in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, d))
                ]
                subdirs.sort(key=lambda x: x[1])

                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        assert isinstance(model, deepspeed.DeepSpeedEngine)
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)

    # 加载检查点
    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        return model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )
