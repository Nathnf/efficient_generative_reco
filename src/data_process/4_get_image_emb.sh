
export CUDA_VISIBLE_DEVICES=0,1

Dataset=Clothing
Data_path=../../data/amazon18
Model_cache_dir=../../cache_models

python clip_feature.py \
    --image_root $Data_path/Images \
    --save_root $Data_path/ \
    --model_cache_dir $Model_cache_dir/clip \
    --dataset $Dataset

