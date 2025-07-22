import datasets
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _parallel_filter_file(self, file_path, context_length, file_type):
        """并行处理单个文件"""
        result = {}
        matched_count = 0
        
        print(f"[{file_type}] 开始处理 {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get('context_length') == context_length:
                        if file_type == "queries":
                            result[data["qid"]] = data["text"]
                        elif file_type == "corpus":
                            result[data["doc_id"]] = {"text": data["text"]}
                        elif file_type == "qrels":
                            result[data["qid"]] = {data["doc_id"]: 1}
                        matched_count += 1
                        
                    if line_num % 20000 == 0:
                        print(f"[{file_type}] 已处理 {line_num} 行，匹配 {matched_count} 行")
                        
                except json.JSONDecodeError:
                    continue
                    
        print(f"[{file_type}] 完成：匹配 {matched_count} 行")
        return file_type, result

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        # 检查是否有本地扩展数据
        local_data_dir = "./local_data/needle"
        use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                         os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                         os.path.exists(f"{local_data_dir}/qrels.jsonl"))

        if use_local_data:
            print(f"使用并行加载needle数据，context_length={context_length}")
            
            # 并行处理三个文件
            file_tasks = [
                (f"{local_data_dir}/queries.jsonl", context_length, "queries"),
                (f"{local_data_dir}/corpus.jsonl", context_length, "corpus"),
                (f"{local_data_dir}/qrels.jsonl", context_length, "qrels")
            ]
            
            results = {}
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=3) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(self._parallel_filter_file, file_path, context_length, file_type): file_type
                    for file_path, context_length, file_type in file_tasks
                }
                
                # 等待结果
                for future in as_completed(future_to_file):
                    file_type, data = future.result()
                    results[file_type] = data
                    print(f"[{file_type}] 线程完成")
            
            queries = results["queries"]
            corpus = results["corpus"]
            qrels = results["qrels"]
            
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

        print(f"并行加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系")

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True 