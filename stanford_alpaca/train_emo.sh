export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Start training"
torchrun --nproc_per_node=4 --master_port=2346 stanford_alpaca/train.py \
    --model_name_or_path /cpfs01/shared/NLP-A100/NLP-A100_hdd/share_model/7b \
    --mode emo \
    --data_path stanford_alpaca/alpaca_data.json \
    --bf16 True \
    --output_dir stanford_alpaca/output_emo_alpaca \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --report_to none \
    --tf32 True