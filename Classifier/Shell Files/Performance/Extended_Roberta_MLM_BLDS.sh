#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G      # memory; default unit is megabytes
#SBATCH --time=10:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
cd /home/partha9/EmbeddingProject/
export CUDA_LAUNCH_BLOCKING=1; python Classifier/TextClassifierPerformance.py \
--model_path /project/def-m2nagapp/partha9/Aster/Classifier_Text_Extended_Roberta_MLM_BLDS/ \
--model_no 9 \
--scratch_path /scratch/partha9/ \
--tokenizer_path /project/def-m2nagapp/partha9/Aster/Text_Extended_Roberta_MLM/ \
--token_max_size 1498 \
--batch_size 32 \
--test_data_path /project/def-m2nagapp/partha9/Aster/
