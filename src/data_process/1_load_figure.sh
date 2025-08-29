Dataset=Clothing
Data_path=../../data/amazon18

python load_all_figures.py \
    --dataset $Dataset \
    --meta_data_path $Data_path/Metadata \
    --rating_data_path $Data_path/Ratings \
    --review_data_path $Data_path/Review \
    --save_path $Data_path/Images