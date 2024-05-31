# PRM Training with Eval

## 1. 基础环境配置

在已经有基本环境（python+pytorch）的机器上，只需要安装如下包：

```shell
pip install deepspeed;pip install transformers;pip install peft;pip install bitsandbytes;pip install datasets;pip install loralib;pip install wandb;
```

## 2. 数据预处理

1. 可参照prm_training_by_step_last_position\data_preprocess.py，主要将原始数据转化为可训练的格式，比如这里由于llama的特殊token是<|reserved_special_token_250|>，所以用这个来作为分步标志。处理后的数据格式应该满足：

   ```json
   {"prompt": "Question: James buys $3000 worth of stuff from Amazon.  He has to return a TV that cost $700 and a bike that cost $500.  He also sells another bike that cost 20% more than the bike he returned for 80% of what he bought it for.  He then buys a toaster for $100.  How much is he out of pocket for everything?\nAnswer: He returns a TV for 700 and a bike for 500 so he returns 700+500 = $1200 worth of stuff<|reserved_special_token_250|>He bought something for 3000 and returns 1200 so that leaves 3000-1200 = $1800<|reserved_special_token_250|>He bought a bike that cost 20% more than the one he returned so he paid 100%-20% = 80% of what the original bike cost<|reserved_special_token_250|>That bike cost 500 and he paid 80% so he paid 80%*500 = $400<|reserved_special_token_250|>He bought the bike for 400 and sells it for 80% of what he paid so he sells it for 400*0.8 = $320<|reserved_special_token_250|>That means he is out of pocket 400-320 = $80<|reserved_special_token_250|>So he is out of pocket 80+1200+1800+100 = $2680 The answer is: 2680<|reserved_special_token_250|>", "label": [0, 0, 0, 0, 0, 0, 0]}
   ```
2. 可参照prm_training_by_step_last_position\data_split_forTrainAndTest.py，主要将预处理侯的数据按照比例分为训练集和测试集。如果是代码开发过程中，可能不需要全量数据，可直接参照prm_training_by_step_last_position\data_split_forCodeTest.py进行指定长度的划分。

一定注意剔除掉无效数据，比如："label": [] 否则会导致后续训练梯度爆炸

```json
{"prompt": "Question: Dana normally drinks a 500 ml bottle of soda each day. Since the 500 ml bottles are currently out of stock at the store, she buys a 2-liter bottle of soda instead. If Dana continues to drink 500 ml of soda each day, how long will the 2-liter bottle of soda last? There are 1,000 ml in 1 liter. So 2 liters is 2 * 1000 = <<2*1000=2000>>2,000 ml. \u043a\u0438\nDana drinks 500 ml of soda each day, so over 2,000 ml, she would drink 500 * 2,000 = <<500*2000=1000>>1,000 ml. \u043a\u0438\nThus, the 2-liter bottle of soda would last Dana 2,000 - 1,000 = <<2000-1000=1000>>1,000 days. The answer is: 1,000 \u043a\u0438\nAnswer: ", "label": []}
```

也一定注意校验每一个prompt里面的special token数量和label长度一致，比如这里应该统计prompt中<|reserved_special_token_250|>的数量和label list 的 length 一样。否则会导致后续训练失败。

## 3. 运行

```shell
./train_rm_llama_test.sh
```

如果执行推理，还需要的安装包有jsonlines

如果考虑warning，可以安装包libaio-dev
