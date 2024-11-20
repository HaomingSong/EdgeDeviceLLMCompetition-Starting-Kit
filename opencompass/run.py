import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'
os.environ["MKL_THREADING_LAYER"] = '1'

#os.environ['TRANSFORMERS_CACHE'] = '/scratch-share/transformers'
os.environ['HF_DATASETS_CACHE'] = '/mnt/hwfile/optimal/LLMComp-Neurips2024/hf_cache'
os.environ['HF_TOKENIZERS_CACHE'] = '/mnt/hwfile/optimal/LLMComp-Neurips2024/hf_cache/tokenizes'
os.environ['HF_HOME'] = '/mnt/hwfile/optimal/LLMComp-Neurips2024/hf_cache/HF_HOME'
os.environ['HF_METRICS_CACHE'] = '/mnt/hwfile/optimal/LLMComp-Neurips2024/hf_cache/metrics'
os.environ['HF_MODULES_CACHE'] = '/mnt/hwfile/optimal/LLMComp-Neurips2024/hf_cache/modules'

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
