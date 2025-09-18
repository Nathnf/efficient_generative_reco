for lr in 1e-2 1e-3; do
    for seed in 1 2 3; do
        for sample in false true; do
            PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=$((2309 + seed)) \
                src/parallel_tiger/test_ddp_tiger.py \
                exp_name="tiger_training_lr${lr}_seed${seed}" \
                infer.do_sample=$sample
        done
    done
done