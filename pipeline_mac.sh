#!/bin/bash

project_path="/Users/dpf/workspace/math-feedback"
num_parallel_process=10
cd "$project_path"

echo "Starting the data preprocessing pipeline..."

cd "${project_path}/utils"
dataset="math_shepherd"
has_label="hasnot"
has_response="hasnot"
input_file_path="${project_path}/raw_data/peiyi9979_Math_Shepherd/math-shepherd.jsonl"
num_points=100
new_folder_suffix="math_shepherd_test_data${num_points}"
language="en"
output_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}.jsonl"
python data_preprocessing.py "$input_file_path" "$output_file_path" "$num_points" "$language" "$dataset" "$has_label" "$has_response"

echo "Data preprocessing completed."

echo "Starting backend generation..."

cd "${project_path}/shepherd_prm"
prompt_template_path="${project_path}/shepherd_prm/templates/criticllm_math_template.txt"
prompt_key="question"
response_key="response"
reference_key="answer"

backbone="tgi"
input_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}.jsonl"
mode="response"
num_parallel_process=$num_parallel_process
url="http://172.18.64.8:8080/generate"

python query_api.py --input_file "$input_file_path" --prompt_template "$prompt_template_path" --prompt_key "$prompt_key" --response_key "$response_key" --reference_key "$reference_key" --backbone "$backbone" --mode "$mode" --num_process "$num_parallel_process" --url "$url"

echo "Backend generation completed."
echo "Starting backend scoring..."

backbone="tgi"
input_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi.jsonl"
mode="critic"
num_parallel_process=$num_parallel_process
url="http://172.18.64.55:8080/generate"

python query_api.py --input_file "$input_file_path" --prompt_template "$prompt_template_path" --prompt_key "$prompt_key" --response_key "$response_key" --reference_key "$reference_key" --backbone "$backbone" --mode "$mode" --num_process "$num_parallel_process" --url "$url"

echo "Backend scoring completed."
echo "Starting forward process path prediction..."

prompt_template_path="${project_path}/shepherd_prm/templates/criticllm_math_template.txt"
prompt_key="question"
response_key="response"
reference_key="answer"
process_response_key="generated_paths"
reference_answer_key="answer"

backbone="tgi"
input_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi_math_critic.jsonl"
mode="generation"
num_parallel_process=$num_parallel_process
url="http://172.18.64.8:8080/generate"

python prm_evaluate_process.py --input_file "$input_file_path" --prompt_template "$prompt_template_path" --prompt_key "$prompt_key" --response_key "$response_key" --reference_key "$reference_key" --process_response_key "$process_response_key" --reference_answer_key "$reference_answer_key" --backbone "$backbone" --mode "$mode" --num_process "$num_parallel_process" --url "$url"

echo "Forward process path prediction completed."
echo "Starting forward process path evaluation..."

backbone="tgi"
input_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi_math_critic_path.jsonl"
mode="critic"
num_parallel_process=$num_parallel_process
url="http://172.18.64.55:8080/generate"

python prm_evaluate_process.py --input_file "$input_file_path" --prompt_template "$prompt_template_path" --prompt_key "$prompt_key" --response_key "$response_key" --reference_key "$reference_key" --process_response_key "$process_response_key" --reference_answer_key "$reference_answer_key" --backbone "$backbone" --mode "$mode" --num_process "$num_parallel_process" --url "$url"

echo "Forward process path evaluation completed."
echo "Calculating accuracy..."

cd "$project_path"

file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi_math_critic_path_math_critic2.jsonl"
output_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi_math_critic_path_math_critic2_statistics.csv"

python Check3_CalculatePathPredictAccuracy.py "$file_path" "$output_file_path"

echo "Accuracy calculation completed."
echo "Starting forward automatic labeling..."

cd "${project_path}/utils"

input_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi_math_critic_path_math_critic2.jsonl"
output_file_path="${project_path}/data/${new_folder_suffix}/front/${new_folder_suffix}.jsonl"
python turn_response_and_solution.py "$input_file_path" "$output_file_path"

cd "$project_path"

source_folder="${project_path}/data/${new_folder_suffix}/front"
target_folder="${project_path}/data/${new_folder_suffix}/front_step1"
python Step1_SplitByRow.py "$source_folder" "$target_folder"

echo "Forward automatic labeling finished Step1_SplitByRow."

source_folder="${project_path}/data/${new_folder_suffix}/front_step1"
target_folder="${project_path}/data/${new_folder_suffix}/front_step2"
python Step2_IsCalculationOrReasoning.py "$source_folder" "$target_folder"

echo "Forward automatic labeling finished Step2_IsCalculationOrReasoning."

source_folder="${project_path}/data/${new_folder_suffix}/front_step2"
target_folder="${project_path}/data/${new_folder_suffix}/front_step3"
backbone="tgi"
url="http://172.18.64.8:8080/generate"
max_words="$num_parallel_process"
python Step3_JudgmentStepCalculatedCorrectly.py "$source_folder" "$target_folder" "$max_words" "$backbone" "$url"

echo "Forward automatic labeling finished Step3_JudgmentStepCalculatedCorrectly."

source_folder="${project_path}/data/${new_folder_suffix}/front_step3"
target_folder="${project_path}/data/${new_folder_suffix}/front_step4"
max_words="$num_parallel_process"
backbone1="tgi"
url1="http://172.18.64.8:8080/generate"
backbone2="tgi"
url2="http://172.18.64.55:8080/generate"
python Step4_JudgmentStepReasoningCorrectly.py "$source_folder" "$target_folder" "$max_words" "$backbone1" "$url1" "$backbone2" "$url2"

echo "Forward automatic labeling finished Step4_JudgmentStepReasoningCorrectly."

echo "Forward automatic labeling completed."

echo "Forward process path evaluation completed."

echo "Calculating accuracy2..."

file_path="${project_path}/data/${new_folder_suffix}/front_Check2Step4/${new_folder_suffix}.jsonl"
output_file_path="${project_path}/data/${new_folder_suffix}/${new_folder_suffix}_tgi_math_critic_path_math_critic2_statistics2.csv"

python Check3_CalculatePathPredictAccuracy.py "$file_path" "$output_file_path"

echo "Accuracy2 calculation completed."

echo "Calculating confusion matrix..."

file_path="${project_path}/data/${new_folder_suffix}/front_Check2Step4/${new_folder_suffix}.jsonl"
output_file_path="${project_path}/data/${new_folder_suffix}/front_Check2Step4/${new_folder_suffix}_ConfusionMatrix.csv"

python Check4_CalculateConfusionMatrix.py "$file_path" "$output_file_path"

echo "Confusion matrix calculation completed."

echo "All steps completed successfully!"
