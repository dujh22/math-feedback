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

如果是在已经有基本环境（python+pytorch）的机器上，只需要安装如下包：

```shell
pip install deepspeed;pip install transformers;pip install peft;pip install bitsandbytes;pip install datasets;pip install loralib;pip install wandb;
```

## 3. 数据预处理

1. 可参照prm_training_by_step_last_position\data_preprocess.py，主要将原始数据转化为可训练的格式，比如这里由于llama的特殊token是<|reserved_special_token_250|>，所以用这个来作为分步标志。处理后的数据格式应该满足：

   ```json
   {"prompt": "Question: James buys $3000 worth of stuff from Amazon.  He has to return a TV that cost $700 and a bike that cost $500.  He also sells another bike that cost 20% more than the bike he returned for 80% of what he bought it for.  He then buys a toaster for $100.  How much is he out of pocket for everything?\nAnswer: He returns a TV for 700 and a bike for 500 so he returns 700+500 = $1200 worth of stuff<|reserved_special_token_250|>He bought something for 3000 and returns 1200 so that leaves 3000-1200 = $1800<|reserved_special_token_250|>He bought a bike that cost 20% more than the one he returned so he paid 100%-20% = 80% of what the original bike cost<|reserved_special_token_250|>That bike cost 500 and he paid 80% so he paid 80%*500 = $400<|reserved_special_token_250|>He bought the bike for 400 and sells it for 80% of what he paid so he sells it for 400*0.8 = $320<|reserved_special_token_250|>That means he is out of pocket 400-320 = $80<|reserved_special_token_250|>So he is out of pocket 80+1200+1800+100 = $2680 The answer is: 2680<|reserved_special_token_250|>", "label": [0, 0, 0, 0, 0, 0, 0]}
   ```
2. 可参照prm_training_by_step_last_position\data_split_forTrainAndTest.py，主要将预处理侯的数据按照比例分为训练集和测试集。如果是代码开发过程中，可能不需要全量数据，可直接参照prm_training_by_step_last_position\data_split_forCodeTest.py进行指定长度的划分。

## 3. 运行

```shell
./train_rm_llama_test.sh
```

如果执行推理，还需要的安装包有jsonlines

如果考虑warning，可以安装包libaio-dev
