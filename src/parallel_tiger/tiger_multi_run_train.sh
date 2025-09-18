for lr in 1e-2 1e-3; do
    for seed in 1 2 3; do
        PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=$((2309 + seed)) \
            src/parallel_tiger/train_tiger.py \
            exp_name="tiger_training_lr${lr}_seed${seed}" \
            seed=$seed \
            train.learning_rate=$lr \
            train.save_and_eval_strategy="steps" \
            train.save_and_eval_steps=50 \
            train.max_steps=12500 \
            train.warmup_steps=1250
    done
done
