
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
python /home/partha9/EmbeddingProject/Roberta/Extended/Electra_PreTrained.py --do_train \
--do_eval \
--train_file train_data.csv \
--dev_file test_data.csv \
--max_seq_length 1498 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--gen_model_name_or_path /project/6033386/partha9/model_cache/roberta_model \
--dis_model_name_or_path /project/6033386/partha9/model_cache/roberta_1500_config \
--tokenizer_name /project/6033386/partha9/model_cache/roberta_tokenizer \
--fp16 \
--output_dir /project/def-m2nagapp/partha9/Aster/Reformer_Electra/