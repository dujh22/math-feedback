# PRM-Inference 采用过程奖励模型进行基本推理

## 1. 环境配置

建议配置：

python==3.10.13

pytorch==2.0.1# py3.10_cuda11.7_cudnn8.5.0_0

然后运行：

```shell
pip install -r requirements.txt
```

## 2. 基本运行

首先请保证PRM模型已被下载到prm_inference\models路径下，下载可采用models_urls.txt记录下载链接，然后使用如下命令进行下载。

```shell
python model_download.py
```

具体运行为

```shell
python inference.py
```

注意修改其中的模型路径

```python
tokenizer = AutoTokenizer.from_pretrained('F:/code/github/math-feedback/math-feedback/prm_inference/models/peiyi9979_math_shepherd_mistral_7b_prm')

model = AutoModelForCausalLM.from_pretrained('F:/code/github/math-feedback/math-feedback/prm_inference/models/peiyi9979_math_shepherd_mistral_7b_prm').eval()
```
