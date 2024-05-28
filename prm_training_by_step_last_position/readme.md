# PRM Training

## 1. 基础环境配置

```shell
conda create -n openrlhf
conda activate openrlhf
conda install python==3.101.4

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
或者
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 2. 适配性环境配置

```shell
cd  ../workspace/dujh22/math_feedback/prm_training_by_step_last_position/
pip install -r requirements.txt
pip install flash-attn==2.5.8
```

deepspeed

tran

peft

bitsandbytes

datasets

loralib

wandb

## 3. 运行

```shell
./train_rm_llama_test.sh

```

如果执行推理，还需要的安装包有

jsonlines



如果考虑warning，可以安装包

libaio-dev
