#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
echo "Starting training..."
/home/jay_agarwal_2022/miniforge3/envs/epignn/bin/python -u mlp_kd_engd.py > ml_kd_engd_depth6.log 2>&1
echo "Training finished. Plotting losses..."
/home/jay_agarwal_2022/miniforge3/envs/epignn/bin/python plot_losses_depth6.py
echo "Running evaluation..."
/home/jay_agarwal_2022/miniforge3/envs/epignn/bin/python eval_mlpkd_engd.py > eval_depth6.log 2>&1
echo "Experiment completed."
