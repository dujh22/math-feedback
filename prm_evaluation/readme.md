## PRM_Evaluation

## 1. 配置

注意，在初始化LlamaTokenizerFast时，如果没有找到可用的tokenizer文件或实例：

运行以下命令安装sentencepiece：

pip install sentencepiece

## 2. 运行

```shell
cd /workspace/dujh22/math_feedback/prm_evaluation
```

### 2.1 数据拆分与response生成

原始数据需要命名为数据集名称.jsonl，放在prm_evaluation\data文件夹下！

#### 2.1.1 数据拆分

将原始数据拆分为8个部分，用于在8卡机上并行生成response

```shell
python prm_evaluation\best_of_nV3_step0_data_split.py 
```

这里一般需要调整的部分是

```python
def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")
    parser.add_argument('--data_length', type=int, default=1319, help='Length of the data')
    parser.add_argument('--num_splits', type=int, default=8, help='Number of splits')
    parser.add_argument('--dataset_name', type=str, default='gsm8k', help='Name of the dataset')
```

表示输出会在输入文件的同级目录下形成一个与数据集名称相同的文件夹内。

#### 2.1.2 response生成

除去基本的python与pytorch，可能需要安装blobfile、transformers、sentencepiece，以及其他所有可能报错缺失的包。

针对每一张显卡，设置显卡号

```python
gpu_id = 0 # 直接在下述文件中修改
```

并运行

```shell
python best_of_nV3_step1_generate_responses_from_mistral.py
```

这里一般需要调整的部分是

```python
def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--max_workers_num', type=int, default=10, help='Maximum number of workers')
    parser.add_argument('--maxn', type=int, default=32, help='Maximum value of n')
    parser.add_argument('--data_length', type=int, default=1319, help='Length of the data')
    parser.add_argument('--generate_backbone', type=str, default="mistral7b", help='Backbone for generation')
    parser.add_argument('--generate_url', type=str, default=TGI_URL, help='URL for generation backbone')
    parser.add_argument('--dataset_name', type=str, default='gsm8k', help='Name of the dataset')
```

### 2.2 数据合并与prm计算

#### 2.2.1 格式转换

首先将数据再次合并并转换为prm模型需要的格式

```shell
python best_of_nV3_step1_1_merge_data_for_prm.py
```

这里一般需要调整的部分是

```python
def main():
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_folder = project_path + dataset_name + '1/'
    output_file = project_path + dataset_name + '1_1/' + dataset_name + '.jsonl'
```

#### 2.2.2 计算PRM

这里需要安装的包有：

```shell
pip install deepspeed;
pip install jsonlines;
pip install peft;
pip install bitsandbytes;
pip install datasets;
```

然后运行获得各response的奖励值

```shell
cd ..
cd ./prm_training_by_step_last_position
./inference_rm_llama_test.sh
```

这里一般需要调整的部分是

```bash
read -r -d '' get_rewards_commands <<EOF
/workspace/dujh22/math_feedback/prm_training_by_step_last_position/inference_rm_llama.py
    --train_data_dir /workspace/dujh22/math_feedback/prm_evaluation/data/gsm8k1_1/gsm8k.jsonl \
    --output_path /workspace/dujh22/math_feedback/prm_evaluation/data/gsm8k1_1/gsm8k_rm.jsonl
EOF
```

#### 2.2.3 数据合并

然后合并回原始数据集中python best_of_nV3_step1_1_merge_data_for_prm.py

```shell
python best_of_nV3_step1_2_merge_data_for_prm.py
```

这里一般需要调整的部分是

```python
def main():
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_folder = project_path + dataset_name + '1/'
    input_file = project_path + dataset_name + '1_1/' + dataset_name + '_rm.jsonl'
    output_file = project_path + dataset_name + '1_1/' + dataset_name + '_rm2.jsonl'
```

注意input_folder包含了原始数据的必要键值对，input_file包含了prm结果，所以需要将两部分进行合并

#### 2.2.4 结果评价

首先测试Critic Model是否可用，这里给的例子是

```shell
cd ./llm
python llm_response.py
```

然后利用Critic Model 对模型答案的正确性进行校验

```shell
python best_of_nV3_step2_calculate_prm_values.py
```

这里一般需要调整的部分是

```python
def prm_evaluation_best_of_n(max_workers_num = 10, data_length = 5, critic_backbone = "tgi", critic_url = CRITIC_URL):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm2.jsonl'
    output_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3.jsonl'

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--max_workers_num', type=int, default=100, help='Maximum number of workers')
    parser.add_argument('--data_length', type=int, default=1319, help='Length of the data')
    parser.add_argument('--critic_backbone', type=str, default="tgi", help='Backbone for critic')
    parser.add_argument('--critic_url', type=str, default=TGI_URL, help='URL for critic backbone')
```

### 2.3 Bon绘图

这里需要安装的包有：

```shell
pip install wandb
```

然后运行

```shell
python best_of_nV3_step3_calculate_accuracy_and_plot.py
```

这里一般需要调整的部分是

```python
def prm_evaluation_best_of_n(maxn = 5, data_length = 5):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3.jsonl'
    output_csv_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3.csv'
    output_pic_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3.png'

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best of N parameterized models.")

    parser.add_argument('--maxn', type=int, default=32, help='Maximum value of n')
    parser.add_argument('--data_length', type=int, default=1319, help='Length of the data')
  
```

### 2.4 中间修正

如果发现其中某步存在数据问题，希望单独运行，那么可以按照下面的说明进行：

这里建议针对上面涉及到的步骤，注意修改文件名称防止覆盖，比如在文件名最后添加V2

#### 2.4.1 修改prm计算

修改prm计算（inference_rm_llama_test.sh）后运行获得各response的奖励值

```shell
cd ..
cd ./prm_training_by_step_last_position
./inference_rm_llama_test.sh
```

然后合并回原始数据集中python best_of_nV3_step1_1_merge_data_for_prm.py

```shell
python best_of_nV3_step1_2_merge_data_for_prm.py
```

然后针对2.2.4 结果评价部分，因为之前已经生成过，请关注best_of_nV3_step2_calculate_prm_values.py中这一部分可以打开注释，注意其中的文件路径是否需要修改

```python
def calculate_prm_values(data, critic_backbone, critic_url, max_workers_num, output_file_path):
    # # 假设已经跑过一次拥有了结果，只是希望合并进来
    # raw_data = []
    # with open("/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/test_rm3.jsonl", 'r', encoding='utf-8') as f0:
    #     for line in f0:
    #         temp_data = json.loads(line.strip())
    #         raw_data.append(temp_data)
    # with open(output_file_path, 'w', encoding='utf-8') as f:
    #     for item in tqdm(data, desc='Calculating N PRMvalue'):
    #         reference_item = next((i for i in raw_data if i.get('unique_id') == item.get('unique_id')), None)
    #         item['extracted_answer'] = reference_item["extracted_answer"]
    #         item['llm_answer_flag'] = reference_item['llm_answer_flag']
    #         item['llm_response_flag'] = reference_item['llm_response_flag']
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    #         f.flush()
    # return data
    # 否则，直接跑
```

然后执行2.3即可。

### 2.5 对比运行

#### 2.5.1 单模型对比

如果希望和其他prm进行对比，首先建议开发一个对应的prm计算的脚本，比如本项目中的prm_inference\inference_mistral7b.py就是针对mistral7b的prm的一个实现，在此基础上，可以开发：

```shell
python best_of_nV3_step4_calculate_another_prm.py
```

这里一般需要调整的部分是

```python
def prm_evaluation_best_of_n(data_length = 5):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3.jsonl'
    output_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3_mathshepherd_prm.jsonl'

def main():
    prm_evaluation_best_of_n(data_length=1319)
```

这个脚本可以在2.3结束之后直接运行，无需返回，运行之后可再次运行2.3即可获得新的prm的绘图。注意运行2.3前修改参数

```python
def prm_evaluation_best_of_n(maxn = 5, data_length = 5):
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_file_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3_mathshepherd_prm.jsonl'
    output_csv_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3_mathshepherd_prm.csv'
    output_pic_path = project_path + dataset_name + '1_1/' + dataset_name + '_rm3_mathshepherd_prm.png'
```

#### 2.5.2 多结果对比

如果希望对上面原始模型和新的单模型进行联合数据对比，可以采用如下脚本

```shell
python best_of_nV3_step5_1_plot_csv_files.py
python best_of_nV3_step5_2_plot_csv_files.py
python best_of_nV3_step5_3_plot_csv_files.py
```

其中5_1是基础折线图绘制，5_2是基础曲线图绘制，5_3是平滑图绘制

注意修改相关参数

```python
if __name__ == "__main__":
    # 指定文件路径
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"
    dir_path = project_path + dataset_name + '1_1/'
    # 创建文件和前缀的映射字典
    file_prefix_mapping = {
        dir_path + 'gsm8k_rm3.csv': 'Train_llama3-8b-instruct-prm(mean)',
        dir_path + 'gsm8k_rm3_mathshepherd_prm.csv': 'Open_Mistral-7b-prm(mean)'
    }
    output_pic_path = dir_path + 'combined_accuracy_plot.png'
```
