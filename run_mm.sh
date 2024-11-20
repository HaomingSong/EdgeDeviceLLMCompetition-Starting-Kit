set -x

OUR_MODEL=true
OUR_MODEL_PATH=/mnt/petrelfs/songhaoming/projs/LLMComp-Neurips2024/crazylm/outputs/crazylm_v2_0_pretrain/2024-08-26/18-33-59_en+zh_tf32_warmup0.01_linear_lr5e-5_bs8_ga1_node1_gpu8/checkpoint-285000
models=(
  # "Qwen2-0.5B"
  # "Qwen2-0.5B-Instruct"
  # "Qwen2-1.5B"
  # "Qwen2-1.5B-Instruct"
  "CrazyLM-1.5B-v1"
)

PARTITION=${PARTITION:-"optimal"}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

for model_name in "${models[@]}"
do
  note=
  cur_date=$(date "+%H-%M-%S")
  date_dir=$(date "+%Y-%m-%d")
  OUTPUT_DIR=outputs/mm/${model_name}/${note:+_$note}/${date_dir}
  mkdir -p ${OUTPUT_DIR}

  if [ "$OUR_MODEL" = true]; then
    model_path=${OUR_MODEL_PATH}
  else
    model_path=/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/${model_name}
  fi


  note=

  sbatch -p ${PARTITION} \
    -J mm_${model_name}${note:+_$note} \
    -o ${OUTPUT_DIR}/%j_${cur_date}.out \
    --gres=gpu:${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=${QUOTA_TYPE} \
    ${SRUN_ARGS} \
    --wrap="python EvaluateThrougthputAndMemory.py \
      --model_name ${model_path}"
done