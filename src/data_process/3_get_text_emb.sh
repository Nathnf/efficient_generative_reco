
export CUDA_VISIBLE_DEVICES=0,1

Dataset=Clothing
Data_path=../../data/amazon18
Model_cache_dir=../../cache_models

python amazon_text_emb.py \
    --dataset $Dataset \
    --root $Data_path \
    --model_cache_dir $Model_cache_dir #--gpu 0,1