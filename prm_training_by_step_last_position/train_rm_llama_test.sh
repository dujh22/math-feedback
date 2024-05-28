set -x 

read -r -d '' training_commands <<EOF
/workspace/dujh22/math_feedback/prm_training_by_step_last_position/train_rm.py \
     --save_path /workspace/dujh22/models/llama3_8B_rw_debug2 \
     --save_steps 50000 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 256 \
     --micro_train_batch_size 1 \
     --pretrain /workspace/dujh22/models/llama3-8B-Instruct \
     --bf16 \
     --max_epochs 2 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --train_data_dir /workspace/dujh22/math_feedback/prm_training_by_step_last_position/raw_data/test/train.jsonl \
     --test_data_dir /workspace/dujh22/math_feedback/prm_training_by_step_last_position/raw_data/test/test.jsonl \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --chosen_key chosen \
     --rejected_key rejected \
     --use_wandb 76ea5b2b06f6f9a718116bb3ec0bd54936f2fded
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi

# if [[ ${1} != "slurm" ]]; then
#     deepspeed --include="localhost:4,5" $training_commands
# fi