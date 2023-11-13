export CUDA_VISIBLE_DEVICES=0,1,2,3

model=$1 # model path for llama
output_dir=$2
mode=emo2
lr=2e-5
blk_size=256
dataset_name=wikitext-103-raw-v1 # dataset used for lightweight fine-tuning

torchrun --nproc_per_node=4 --master_port=2346 run_clm_trainer_emo_fsdp.py \
    --model_name_or_path $model \
    --mode $mode \
    --dataset_name wikitext \
    --dataset_config_name $dataset_name \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --preprocessing_num_workers 32 \
    --block_size $blk_size \
    --seed 100 \
    --learning_rate $lr \
    --do_train \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --warmup_ratio 0.03 \
    --report_to none \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True