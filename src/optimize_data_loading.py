#!/usr/bin/env python3
"""
优化数据加载：按context_length预先分割数据文件
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm

def split_data_by_context_length(data_dir="./local_data"):
    """按context_length分割数据文件，提高加载效率"""
    
    for task in ["needle", "passkey"]:
        task_dir = os.path.join(data_dir, task)
        if not os.path.exists(task_dir):
            continue
            
        print(f"\n=== 处理 {task} 任务数据 ===")
        
        # 创建按长度分组的目录
        split_dir = os.path.join(task_dir, "split_by_length")
        os.makedirs(split_dir, exist_ok=True)
        
        # 处理每种数据类型
        for file_type in ["queries", "corpus", "qrels"]:
            file_path = os.path.join(task_dir, f"{file_type}.jsonl")
            if not os.path.exists(file_path):
                continue
                
            print(f"处理 {file_type}.jsonl...")
            
            # 按context_length分组数据
            length_groups = defaultdict(list)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"读取{file_type}"):
                    data = json.loads(line.strip())
                    context_length = data.get('context_length')
                    if context_length is not None:
                        length_groups[context_length].append(data)
            
            # 写入分组文件
            for context_length, items in length_groups.items():
                output_file = os.path.join(split_dir, f"{file_type}_{context_length}.jsonl")
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"  创建 {file_type}_{context_length}.jsonl: {len(items)} 条记录")

def create_optimized_loading_class():
    """创建优化后的数据加载类"""
    
    needle_content = '''import datasets
import os
import json
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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

        # 优化1: 检查是否有按长度分割的文件
        local_data_dir = "./local_data/needle"
        split_dir = os.path.join(local_data_dir, "split_by_length")
        
        use_split_data = (os.path.exists(f"{split_dir}/queries_{context_length}.jsonl") and
                         os.path.exists(f"{split_dir}/corpus_{context_length}.jsonl") and
                         os.path.exists(f"{split_dir}/qrels_{context_length}.jsonl"))
        
        if use_split_data:
            print(f"使用预分割数据加载needle数据，context_length={context_length}")
            # 直接加载特定长度的文件，无需filter
            query_list = datasets.Dataset.from_json(f"{split_dir}/queries_{context_length}.jsonl")
            corpus_list = datasets.Dataset.from_json(f"{split_dir}/corpus_{context_length}.jsonl")
            qrels_list = datasets.Dataset.from_json(f"{split_dir}/qrels_{context_length}.jsonl")
            
            # 直接构建字典，无需filter
            queries = {row["qid"]: row["text"] for row in query_list}
            corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}
            qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}
            
        else:
            # 优化2: 检查是否有本地原始数据
            use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                             os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                             os.path.exists(f"{local_data_dir}/qrels.jsonl"))

            if use_local_data:
                print(f"使用本地数据加载needle数据，context_length={context_length}")
                # 优化3: 流式处理，避免加载整个文件
                queries = {}
                corpus = {}
                qrels = {}
                
                # 流式读取queries
                with open(f"{local_data_dir}/queries.jsonl", 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get("context_length") == context_length:
                            queries[data["qid"]] = data["text"]
                
                # 流式读取corpus
                with open(f"{local_data_dir}/corpus.jsonl", 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get("context_length") == context_length:
                            corpus[data["doc_id"]] = {"text": data["text"]}
                
                # 流式读取qrels
                with open(f"{local_data_dir}/qrels.jsonl", 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get("context_length") == context_length:
                            qrels[data["qid"]] = {data["doc_id"]: 1}
                            
            else:
                print(f"使用HuggingFace数据加载needle数据，context_length={context_length}")
                # 从HuggingFace加载
                query_list = datasets.load_dataset(**self.metadata.dataset)["queries"]
                corpus_list = datasets.load_dataset(**self.metadata.dataset)["corpus"]
                qrels_list = datasets.load_dataset(**self.metadata.dataset)["qrels"]

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
    
    with open("LongEmbed/src/LEMBNeedleRetrieval_optimized.py", 'w', encoding='utf-8') as f:
        f.write(needle_content)
    print("已创建优化版本的LEMBNeedleRetrieval_optimized.py")

if __name__ == "__main__":
    print("开始优化数据加载...")
    
    # 第一步：分割数据文件
    split_data_by_context_length()
    
    # 第二步：创建优化的加载类
    create_optimized_loading_class()
    
    print("\n优化完成！现在可以使用更快的数据加载了。") 