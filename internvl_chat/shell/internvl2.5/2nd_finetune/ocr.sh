set -x

GPUS=1
BATCH_SIZE=8
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='grpo_train_outputs/vintern1b_ocr_2612'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
  
CUDA_VISIBLE_DEVICES=1 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_GRPO_finetune.py \
  --model_name_or_path "/data/llm/Vintern-7B/internvl_chat/model_outputs/vintern-1b-v3_5-ocr" \
  --conv_style "Hermes-2" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/ocr.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 4 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --use_backbone_lora 16 \
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
  --weight_decay 0.0001 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --logging_steps 1 \
  --max_seq_length 1500 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage2_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
