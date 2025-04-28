set -x

GPUS=1
BATCH_SIZE=8
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='grpo_train_outputs/InternVL3-1B-qvq-merged-GRPO'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16 OpenGVLab/InternVL2_5-1B-MPO 
# epoch: 1
  # --use_vllm True \
  # --num_generations 4 \
  # --max_prompt_length 500 \
  # --max_completion_length 1000 \
  # --beta 0.04 \
  # --vllm_gpu_memory_utilization 0.15 \
  # --vllm_max_token 1000 \
  # --temperature 0.5 \
  # --vllm_device "auto" \
  # --max_grad_norm 0.1 \
  # --epsilon_high 0.2 \
  # --epsilon_low 0.2 \
  
CUDA_VISIBLE_DEVICES=1 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_GRPO_finetune.py \
  --model_name_or_path "/data/llm/R1/InternVL/internvl_chat/model_outputs/InternVL3-1B-qvq-merged" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/math_r1.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 4 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 1 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --logging_steps 1 \
  --max_seq_length 2000 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage2_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
