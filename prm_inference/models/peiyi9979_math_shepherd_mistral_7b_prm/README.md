# [Math-Shepherd](https://arxiv.org/pdf/2312.08935.pdf) 中使用的过程奖励模型（mistral-7b）

`Input`: question + step-by-step solutions with a special step tag `ки`, e.g.,

输入"：问题 + 带有特殊步骤标签 ки 的分步解决方案，例如：

```shell
Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes .... ? Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки
```
```shell
珍妮特（Janet）的鸭子每天下 16 个蛋。她每天早餐吃三个，然后烘焙 ....。? 步骤 1：珍妮特的鸭子每天下 16 个蛋。ки 第 2 步：她每天早餐吃 3 个鸡蛋，所以还剩下 16 - 3 = 13 个鸡蛋。ки 第 3 步：她每天用 4 个鸡蛋为朋友烤松饼，所以她还剩下 13 - 4 = 9 个鸡蛋。ки 第 4 步：她每天在农贸市场以每个新鲜鸭蛋 2 美元的价格出售剩余的鸡蛋，因此她每天在农贸市场赚取 9 * 2 = 18 美元。答案是：18 ки
```

`Output`: the logits. You need to post-process it to achieve the score of each step.

输出"：对数。您需要对其进行后处理，以获得每一步的得分。


```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

good_token = '+'
bad_token = '-'
step_tag = 'ки'

tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong

for output in [output1, output2]:
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        step_scores = scores[input_id == step_tag_id]
        print(step_scores)
        
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240]) 

```