#!/bin/bash
#SBATCH --job-name=simplify
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000MB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=./logs/slurm-%A.out

export TOKENIZERS_PARALLELISM=true

source /home/AttanasioG/.bashrc
conda activate py310

MODEL=$1
DATASET=$2

if [ -z "$MODEL" ]
then
    echo "Please specify a model name"
    exit 1
fi

if [ -z "$DATASET" ]
then
    echo "Please specify a dataset name"
    exit 1
fi

python simplify.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --load_in_8bit="true" --load_in_4bit="false" \
    --output_dir ./data/answers/${DATASET}/${MODEL} \
    --do_sample="true" \
    --temperature="0.7" \
    --top_p="1.0" \
    --max_new_tokens="512" \
    --prompt_template="0"
    