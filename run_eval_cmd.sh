set -x
models=(
  "internlm2-chat-1_8b"
  # "chatglm3-6b"
  # "internlm2-chat-1_8b"
  # "internlm2-chat-7b"
  # "Nanbeige2-8B-Chat"
  # "Phi-3-medium-128k-instruct"
  # "Qwen1.5-7B-Chat"
  # "Qwen2-7B"
  # "Yi-1.5-6B-Chat" 
  # "Yi-1.5-9B-Chat"
)

datasets=(
  # "commonsenseqa_gen"
  # "bbh_gen"
  # "gsm8k_gen"
  # "longbench"
  # "humaneval_gen"
  # "truthfulqa_gen"
  "FewCLUE_chid_gen"
)

PARTITION=${PARTITION:-"optimal"}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

datasets_args=$(IFS=" "; echo "${datasets[*]}")
datasets_pth=$(IFS=-; echo "${datasets[*]}")
for model_name in "${models[@]}"
do
  note=
  cur_date=$(date "+%H-%M-%S")
  date_dir=$(date "+%Y-%m-%d")
  echo ${date_dir}
  
  OUTPUT_DIR=outputs/${model_name}${note:+_$note}/${date_dir}
  model_path=/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/${model_name}
  mkdir -p ${OUTPUT_DIR}

  sbatch -p ${PARTITION} \
    -J eval_${model_name}${note:+_$note} \
    -o ${OUTPUT_DIR}/%j_${datasets_pth}_${cur_date}.out \
    --gres=gpu:${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=${QUOTA_TYPE} \
    ${SRUN_ARGS} \
    --wrap="cd opencompass && python run.py \
      --datasets ${datasets_args} \
      --batch-size 8 \
      --hf-num-gpus ${GPUS_PER_NODE} \
      --hf-type base \
      --hf-path ${model_path} \
      --debug \
      --model-kwargs device_map=\'auto\' \
      trust_remote_code=True"
done

# python -m pdb run.py --datasets commonsenseqa_gen --hf-num-gpus 1 --hf-type base \
#   --hf-path /mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Yi-1.5-6B-Chat-6b \
#   --debug --model-kwargs device_map=\'auto\' trust_remote_code=True

# CUDA_VISIBLE_DEVICES=0 python run.py \
#   --datasets commonsenseqa_gen longbench bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen truthfulqa_gen \
#   --hf-num-gpus 1 \
#   --hf-type base \
#   --hf-path microsoft/phi-2 \
#   --debug \
#   --model-kwargs device_map='auto' trust_remote_code=True
