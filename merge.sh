# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-1_7b_mixce_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-1_7b_mixce_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-25K-mixce

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_7b_mixce_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_7b_mixce_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-50K-mixce

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_7b_mixce_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_7b_mixce_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-100K-mixce

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_600K_llama-2_7b_mle_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_600K_llama-2_7b_mle_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-600K-mle

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_600K_llama-2_7b_tvd_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_600K_llama-2_7b_tvd_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-600K-tvd

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-50K-onehot_emd2

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-100K-onehot_emd2

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_200K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_200K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-200K-onehot_emd2

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_200K_llama-1_13b_emd_only_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_200K_llama-1_13b_emd_only_1.0_lr2e-5_256 \
#             --output_dir ./llama-1-13b-wiki103-200K-emdonly

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_mle_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_mle_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-13b-wiki103-25K-mle

python merge_lora.py \
            --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_7b_emo_1.0_lr2e-5_256 \
            --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_7b_emo_1.0_lr2e-5_256 \
            --output_dir ./llama-2-7b-wiki103-25K-emo-13bhead

python merge_lora.py \
            --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_7b_emo_1.0_lr2e-5_256 \
            --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_7b_emo_1.0_lr2e-5_256 \
            --output_dir ./llama-2-7b-wiki103-100K-emo-13bhead

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_7b_onehot_emd_2_0.15_lr2e-5_256_new \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_7b_onehot_emd_2_0.15_lr2e-5_256_new \
#             --output_dir ./llama-2-7b-wiki103-100K-onehot_emd2_0.15_new

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256_embedtrainable \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_7b_onehot_emd_2_1.0_lr2e-5_256_embedtrainable \
#             --output_dir ./llama-2-7b-wiki103-25K-onehot_emd2

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_13b_tvd_0.1_lr2e-5_256_alllora \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-2_13b_tvd_0.1_lr2e-5_256_alllora \
#             --output_dir ./llama-2-13b-wiki103-100K-tvd0.1

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_13b_tvd_0.1_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_13b_tvd_0.1_lr2e-5_256 \
#             --output_dir ./llama-1-13b-wiki103-100K-tvd0.1

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-1_7b_tvd_0.1_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-1_7b_tvd_0.1_lr2e-5_256 \
#             --output_dir ./llama-1-7b-wiki103-25K-tvd0.1

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_mixce_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_mixce_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-13b-wiki103-25K-mixce

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_mixce_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_mixce_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-13b-wiki103-25K-mixce

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_tvd_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-2_13b_tvd_1.0_lr2e-5_256 \
#             --output_dir ./llama-2-13b-wiki103-25K-tvd

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_600K_llama-1_13b_mle_1.0_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_600K_llama-1_13b_mle_1.0_lr2e-5_256 \
#             --output_dir ./llama-1-13b-wiki103-600K-mle

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_13b_mixce_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_13b_mixce_0.9_lr2e-5_256 \
#             --output_dir ./llama-1-13b-wiki103-50K-mixce

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_13b_tvd_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_13b_tvd_0.9_lr2e-5_256 \
#             --output_dir ./llama-1-13b-wiki103-50K-tvd

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-1_7b_tvd_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_25K_llama-1_7b_tvd_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-25K-tvd

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_7b_tvd_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_50K_llama-1_7b_tvd_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-50K-tvd

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_7b_tvd_0.9_lr2e-5_256 \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_7b_tvd_0.9_lr2e-5_256 \
#             --output_dir ./llama-2-7b-wiki103-100K-tvd

# python merge_lora.py \
#             --lora_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_7b_tvd_0.9_lr2e-5_256_real \
#             --base_model_name_or_path /cpfs01/user/rensiyu/language_modeling/output/trainer_wiki103_100K_llama-1_7b_tvd_0.9_lr2e-5_256_real \
#             --output_dir ./llama-1-7b-wiki103-100K-tvd
