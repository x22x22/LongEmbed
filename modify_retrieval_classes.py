#!/usr/bin/env python3
"""
修改LEMBNeedleRetrieval和LEMBPasskeyRetrieval类，支持从本地数据加载65536长度数据
"""

def modify_needle_retrieval():
    """修改LEMBNeedleRetrieval.py"""
    
    content = '''import datasets
import os
from mteb.abstasks.TaskMetadata import TaskMetadata
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBNeedleRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBNeedleRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "needle",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("needle subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2023-12-31"),
        form=["written"],
        domains=["Academic", "Blog"],
        task_subtypes=["Article retrieval"],
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={_EVAL_SPLIT: 1200},
        avg_character_length={_EVAL_SPLIT: 35305.2},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        # 检查是否有本地扩展数据（支持65536长度）
        local_data_dir = "./local_data/needle"
        use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                         os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                         os.path.exists(f"{local_data_dir}/qrels.jsonl"))

        if use_local_data:
            print(f"使用本地数据加载needle数据，context_length={context_length}")
            # 从本地JSONL文件加载
            query_list = datasets.Dataset.from_json(f"{local_data_dir}/queries.jsonl")
            corpus_list = datasets.Dataset.from_json(f"{local_data_dir}/corpus.jsonl")
            qrels_list = datasets.Dataset.from_json(f"{local_data_dir}/qrels.jsonl")
        else:
            print(f"使用HuggingFace数据加载needle数据，context_length={context_length}")
            # 从HuggingFace加载
            query_list = datasets.load_dataset(**self.metadata_dict["dataset"])["queries"]
            corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])["corpus"]
            qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])["qrels"]

        # 过滤指定长度的数据
        query_list = query_list.filter(lambda x: x["context_length"] == context_length)
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = corpus_list.filter(
            lambda x: x["context_length"] == context_length
        )
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = qrels_list.filter(lambda x: x["context_length"] == context_length)
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        print(f"加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系")

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
'''
    
    with open("src/LEMBNeedleRetrieval.py", "w") as f:
        f.write(content)
    print("✓ 已修改 LEMBNeedleRetrieval.py")

def modify_passkey_retrieval():
    """修改LEMBPasskeyRetrieval.py"""
    
    content = '''import datasets
import os
from mteb.abstasks.TaskMetadata import TaskMetadata
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBPasskeyRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBPasskeyRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "passkey",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("passkey subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2023-12-31"),
        form=["written"],
        domains=["Fiction"],
        task_subtypes=["Article retrieval"],
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={_EVAL_SPLIT: 1200},
        avg_character_length={_EVAL_SPLIT: 28994.8},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        # 检查是否有本地扩展数据（支持65536长度）
        local_data_dir = "./local_data/passkey"
        use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                         os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                         os.path.exists(f"{local_data_dir}/qrels.jsonl"))

        if use_local_data:
            print(f"使用本地数据加载passkey数据，context_length={context_length}")
            # 从本地JSONL文件加载
            query_list = datasets.Dataset.from_json(f"{local_data_dir}/queries.jsonl")
            corpus_list = datasets.Dataset.from_json(f"{local_data_dir}/corpus.jsonl")
            qrels_list = datasets.Dataset.from_json(f"{local_data_dir}/qrels.jsonl")
        else:
            print(f"使用HuggingFace数据加载passkey数据，context_length={context_length}")
            # 从HuggingFace加载
            query_list = datasets.load_dataset(**self.metadata_dict["dataset"])["queries"]
            corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])["corpus"]
            qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])["qrels"]

        # 过滤指定长度的数据
        query_list = query_list.filter(lambda x: x["context_length"] == context_length)
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = corpus_list.filter(
            lambda x: x["context_length"] == context_length
        )
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = qrels_list.filter(lambda x: x["context_length"] == context_length)
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        print(f"加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系")

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
'''
    
    with open("src/LEMBPasskeyRetrieval.py", "w") as f:
        f.write(content)
    print("✓ 已修改 LEMBPasskeyRetrieval.py")

def backup_original_files():
    """备份原始文件"""
    
    import shutil
    
    files_to_backup = [
        "src/LEMBNeedleRetrieval.py",
        "src/LEMBPasskeyRetrieval.py"
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            print(f"✓ 备份 {file_path} -> {backup_path}")

def modify_scripts_for_65536():
    """修改评估脚本，添加65536支持"""
    
    import re
    
    scripts_to_modify = [
        "scripts/run_long_embed.sh",
        "scripts/run_openai_long_embed.sh"
    ]
    
    for script_path in scripts_to_modify:
        if os.path.exists(script_path):
            # 读取文件内容
            with open(script_path, 'r') as f:
                content = f.read()
            
            # 替换window_length_list，添加65536
            old_pattern = r'--window_length_list 256 512 1024 2048 4096 8192 16384 32768'
            new_pattern = '--window_length_list 256 512 1024 2048 4096 8192 16384 32768 65536'
            
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                
                # 写回文件
                with open(script_path, 'w') as f:
                    f.write(content)
                
                print(f"✓ 已修改 {script_path}，添加65536支持")
            else:
                print(f"⚠ 在 {script_path} 中未找到expected pattern")
        else:
            print(f"⚠ 文件 {script_path} 不存在")

if __name__ == "__main__":
    import os
    
    print("开始修改LEMBNeedleRetrieval和LEMBPasskeyRetrieval类...")
    
    # 备份原始文件
    backup_original_files()
    
    # 修改数据加载类
    modify_needle_retrieval()
    modify_passkey_retrieval()
    
    # 修改评估脚本
    modify_scripts_for_65536()
    
    print("\n✓ 所有修改完成！")
    print("\n接下来的步骤：")
    print("1. 运行 python generate_65536_data.py 生成65536长度的数据")
    print("2. 运行评估脚本测试65536长度的支持")
    print("3. 如果有问题，可以通过 .backup 文件恢复原始版本") 