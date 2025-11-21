import json
import os
import pdb

import datasets


_BASE_URL = "https://huggingface.co/datasets/AILab-CVC/SEED-Bench/raw/main/SEED-Bench.json"


class SEEDBenchDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description="SEED-Bench dataset"),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "answer": datasets.Value("string"),
                    "choice_a": datasets.Value("string"),
                    "choice_b": datasets.Value("string"),
                    "choice_c": datasets.Value("string"),
                    "choice_d": datasets.Value("string"),
                    "data_id": datasets.Value("string"),
                    "data_type": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question_type_id": datasets.Value("string"),
                    "segment": datasets.Value("string", id=None),  # 将此设置为可选字段
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {"data_file": _BASE_URL}
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        # 为数据集创建一个测试拆分
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_file": downloaded_files["data_file"]},
            ),
        ]

    def _generate_examples(self, data_file):
        # pdb.set_trace()
        with open(data_file, 'r') as f:
            data = json.load(f)

        for item in data["questions"]:
            # 提取数据集中的每个字段
            answer = item["answer"]
            choice_a = item["choice_a"]
            choice_b = item["choice_b"]
            choice_c = item["choice_c"]
            choice_d = item["choice_d"]
            data_id = item["data_id"]
            data_type = item["data_type"]
            question = item["question"]
            question_id = item["question_id"]
            question_type_id = item["question_type_id"]
            segment = item.get("segment", None)  # 如果存在 "segment" 字段，则获取其值，否则设置为 None

            # 返回一个包含数据集特征的字典
            yield question_id, {
                "answer": answer,
                "choice_a": choice_a,
                "choice_b": choice_b,
                "choice_c": choice_c,
                "choice_d": choice_d,
                "data_id": data_id,
                "data_type": data_type,
                "question": question,
                "question_id": question_id,
                "question_type_id": question_type_id,
                "segment": segment,
            }