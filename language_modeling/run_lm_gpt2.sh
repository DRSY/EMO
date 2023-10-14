export CUDA_VISIBLE_DEVICES=0,

model_name=gpt2
dataset_name=wikitext-2-raw-v1
bsz=32
block_size=256
decode_newlen=80
decoding_mode=unbiased
epochs=3
mode=emo
seed=42

# hyper-parameters
args=(
    --per_device_train_batch_size $bsz 
    --num_train_epochs $epochs
    --preprocessing_num_workers 16
    --gradient_accumulation_steps 1
    --dataset_name wikitext
    --dataset_config_name $dataset_name
    --model_name_or_path $model_name
    --output_dir ./output/${model_name}_${mode}_${dataset_name}
    --block_size $block_size 
    --seed $seed
    --learning_rate 5e-5
    --decode_newlen $decode_newlen
    --mixing_ratio $mixing_ratio
    --mode $mode
    --decoding_mode $decoding_mode
)

python -u -Wignore run_lm.py ${args[@]} 2>&1 | tee ./logs/${model_name}_${mode}_seed${seed}_${dataset_name}.log