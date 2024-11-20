set -x

PARTITION=${PARTITION:-"optimal"}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

note=
date_dir=$(date "+%Y-%m-%d")
cur_date=$(date "+%H-%M-%S")
echo ${date_dir}

OUTPUT_DIR=outputs/score/${note:+_$note}/${date_dir}
mkdir -p ${OUTPUT_DIR}

sbatch -p ${PARTITION} \
  -J eval_LLM_${note:+_$note} \
  -o ${OUTPUT_DIR}/%j_${cur_date}.out \
  --gres=gpu:${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  --wrap="cd opencompass && \
    python run.py /mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/crazy_llm.py \
    --debug \
    -l"