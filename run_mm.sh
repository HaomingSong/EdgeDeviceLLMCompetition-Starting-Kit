set -x

models=(
  "internlm2-chat-1_8b"
  "chatglm3-6b"
  "internlm2-chat-1_8b"
  "internlm2-chat-7b"
  "Nanbeige2-8B-Chat"
  "Phi-3-medium-128k-instruct"
  "Qwen1.5-7B-Chat"
  "Qwen2-7B"
  "Yi-1.5-6B-Chat" 
  "Yi-1.5-9B-Chat"
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
  model_path=/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/${model_name}
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