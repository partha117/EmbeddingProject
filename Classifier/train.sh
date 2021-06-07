#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G      # memory; default unit is megabytes
#SBATCH --time=18:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/EmbeddingProject/Classifier/TransformerClassifierElectra.py --do_train \
--root_path /project/def-m2nagapp/partha9/Aster/ClassifierReformer \
--model_path /project/def-m2nagapp/partha9/Aster/Reformer_Electra/ \
--project_name /project/def-m2nagapp/partha9/Dataset/CombinedData/ \
--scratch_path /scratch/partha9/Dataset/ \
--tokenizer_root /project/def-m2nagapp/partha9/Aster/Reformer_Electra/ \
--token_max_size 2048 \
--batch_size 64 \
--model_name reformer
