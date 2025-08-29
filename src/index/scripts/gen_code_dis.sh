Dataset=Arts
Quantizer_type=rqvae
Code_num=256
Use_linear=0

OUTPUT_DIR=../../data/amazon18/$Dataset
Model=llama
CKPT_DIR=log/$Dataset/${Quantizer_type}/${Model}_${Code_num}_use_linear_$Use_linear

if [ $Use_linear -eq 1 ]; then
  suffix="_use_linear"
else
  suffix=""
fi

python -u generate_indices_distance.py \
  --quantizer_type $Quantizer_type \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path $CKPT_DIR/best_collision_model.pth \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb_${Quantizer_type}_${Code_num}${suffix}.json \


Model=ViT-L-14
CKPT_DIR=log/$Dataset/${Quantizer_type}/${Model}_${Code_num}_use_linear_$Use_linear

python -u generate_indices_distance.py \
    --quantizer_type $Quantizer_type \
    --dataset $Dataset \
    --device cuda:0 \
    --ckpt_path $CKPT_DIR/best_collision_model.pth \
    --output_dir $OUTPUT_DIR \
    --output_file ${Dataset}.index_vitemb_${Quantizer_type}_${Code_num}${suffix}.json \
    --content image


