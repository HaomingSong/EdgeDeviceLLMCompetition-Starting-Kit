from mmengine.config import read_base

with read_base():
    from .datasets.commonsenseqa.commonsenseqa_7shot_cot_gen_734a22 import commonsenseqa_datasets
    from .datasets.bbh.bbh_gen import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.longbench.longbench import longbench_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets
    from .datasets.FewCLUE_chid.FewCLUE_chid_gen import chid_datasets
    from .models.crazylm.crazylm_1_5b import models

lark_bot_url = "https://open.feishu.cn/open-apis/bot/v2/hook/598e368a-2472-45d1-81b8-434b897c97ce"


datasets = commonsenseqa_datasets + bbh_datasets + gsm8k_datasets + humaneval_datasets + truthfulqa_datasets + chid_datasets # + longbench_datasets