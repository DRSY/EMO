export CUDA_VISIBLE_DEVICES=0,

mode=emo2 # training objective
model=TheBloke/Llama-2-7B-fp16 # model path for llama
bsz=16 # per device training batch size
blk_size=256 # sequence length
lr=2e-5 # learning rate
dataset_name=wikitext-103-raw-v1 # dataset used for lightweight fine-tuning

# we use accelerate as the program launcher
accelerate launch --config_file accelerate_configs/accelerate_config_0.yaml run_clm_trainer_emo.py \
        --model_name_or_path $model \
        --mode $mode \
        --dataset_name wikitext \
        --dataset_config_name $dataset_name \
        --per_device_train_batch_size $bsz \
        --per_device_eval_batch_size 16 \
        --preprocessing_num_workers 32 \
        --block_size $blk_size \
        --seed 100 \
        --learning_rate $lr \
        --num_train_epochs 1 \
        --do_train \
        --output_dir ./output/${dataset_name}_llama_7b_${mode}_lr${lr}_${blk_size} \
        --overwrite_output_dir \
        --mixing_ratio 1.0 \
        --warmup_ratio 0.03 \
        --fp16