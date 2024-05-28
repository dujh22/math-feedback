set -x 

read -r -d '' training_commands <<EOF
/workspace/dujh22/math_feedback/prm_training/train_rm.py \
     --save_path /workspace/dujh22/models/llama3_8B_rw \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain /workspace/dujh22/models/llama3-8B-Instruct \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --train_data_dir /workspace/dujh22/math_feedback/prm_training/raw_data/test/train.jsonl \
     --test_data_dir /workspace/dujh22/math_feedback/prm_training/raw_data/test/test.jsonl \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --chosen_key chosen \
     --rejected_key rejected \
     --use_wandb True
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --include="localhost:3,4" $training_commands
fi
