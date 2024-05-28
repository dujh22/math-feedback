set -x

read -r -d '' get_rewards_commands <<EOF
/workspace/dujh22/math_feedback/prm_training_by_step_last_position/inference_rm_llama.py
    --pretrain /workspace/dujh22/models/llama3_8B_rw_debug2 \
    --bf16 \
    --max_len 2048 \
    --train_data_dir /workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/test.jsonl \
    --dataset_probs 1 \
    --zero_stage 0 \
    --output_path /workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/test_rm.jsonl
EOF

deepspeed $get_rewards_commands