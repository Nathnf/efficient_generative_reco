Quantizer_type=rqvae
Model=llama
Code_num=256
Beta=0.25
Quant_loss_weight=1
Use_linear=1
# Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
Datasets='Arts'


OUTPUT_DIR=log/$Datasets/${Quantizer_type}/${Model}_${Code_num}_use_linear_$Use_linear
mkdir -p $OUTPUT_DIR


if [ $Quantizer_type = "rqvae" ]; then
  sk_epsilons="0.0 0.0 0.0 0.003"
  layers="2048 1024 512 256 128 64"
elif [ $Quantizer_type = "pqvae" ]; then
  sk_epsilons="0.001 0.001 0.001 0.001"
  layers="512 256 128 64"
fi

python -u main_mul.py \
  --quantizer_type $Quantizer_type \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons $sk_epsilons \
  --layers $layers \
  --device cuda:0 \
  --data_root ../../data/amazon18/ \
  --embedding_file .emb-llama-td.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 5 \
  --batch_size 2048 \
  --beta $Beta \
  --quant_loss_weight $Quant_loss_weight \
  --use_linear $Use_linear \
  --epochs 500 > $OUTPUT_DIR/train.log


Model=ViT-L-14

OUTPUT_DIR=log/$Datasets/${Quantizer_type}/${Model}_${Code_num}_use_linear_$Use_linear
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --quantizer_type $Quantizer_type \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons $sk_epsilons \
  --layers $layers \
  --device cuda:0 \
  --data_root ../../data/amazon18/ \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 5 \
  --batch_size 2048 \
  --beta $Beta \
  --quant_loss_weight $Quant_loss_weight \
  --use_linear $Use_linear \
  --epochs 500 > $OUTPUT_DIR/train.log