#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3G      # memory; default unit is megabytes
#SBATCH --time=4:20:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
python /home/partha9/EmbeddingProject/Get_Data.py