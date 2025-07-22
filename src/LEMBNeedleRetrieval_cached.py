import datasets
import os
import json
import pickle
import hashlib
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBNeedleRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    _CACHE_DIR = "./cache/needle"

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

    def _get_cache_path(self, context_length):
        """获取缓存文件路径"""
        os.makedirs(self._CACHE_DIR, exist_ok=True)
        return os.path.join(self._CACHE_DIR, f"needle_data_{context_length}.pkl")

    def _get_file_hash(self, file_path):
        """获取文件的哈希值，用于检测文件是否更改"""
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_from_cache(self, context_length):
        """从缓存加载数据"""
        cache_path = self._get_cache_path(context_length)
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            # 检查缓存版本和文件哈希
            local_data_dir = "./local_data/needle"
            if os.path.exists(f"{local_data_dir}/corpus.jsonl"):
                current_hash = self._get_file_hash(f"{local_data_dir}/corpus.jsonl")
                if cached_data.get('file_hash') != current_hash:
                    print("检测到数据文件已更改，缓存失效")
                    return None
                    
            print(f"从缓存加载数据: context_length={context_length}")
            return cached_data['data']
            
        except Exception as e:
            print(f"缓存加载失败: {e}")
            return None

    def _save_to_cache(self, context_length, data):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(context_length)
        
        # 获取文件哈希用于版本控制
        local_data_dir = "./local_data/needle"
        file_hash = None
        if os.path.exists(f"{local_data_dir}/corpus.jsonl"):
            file_hash = self._get_file_hash(f"{local_data_dir}/corpus.jsonl")
            
        cached_data = {
            'data': data,
            'file_hash': file_hash,
            'context_length': context_length
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"数据已缓存: {cache_path}")
        except Exception as e:
            print(f"缓存保存失败: {e}")

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        # 尝试从缓存加载
        cached_data = self._load_from_cache(context_length)
        if cached_data:
            queries, corpus, qrels = cached_data
            print(f"缓存加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系")
        else:
            # 缓存未命中，重新加载数据
            local_data_dir = "./local_data/needle"
            use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                             os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                             os.path.exists(f"{local_data_dir}/qrels.jsonl"))

            if use_local_data:
                print(f"从本地数据加载needle数据，context_length={context_length}")
                # 流式处理，避免加载整个文件
                queries = {}
                corpus = {}
                qrels = {}
                
                # 流式读取并过滤
                for file_type, data_dict in [("queries", queries), ("corpus", corpus), ("qrels", qrels)]:
                    file_path = f"{local_data_dir}/{file_type}.jsonl"
                    print(f"处理 {file_type}.jsonl...")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if data.get("context_length") == context_length:
                                    if file_type == "queries":
                                        queries[data["qid"]] = data["text"]
                                    elif file_type == "corpus":
                                        corpus[data["doc_id"]] = {"text": data["text"]}
                                    elif file_type == "qrels":
                                        qrels[data["qid"]] = {data["doc_id"]: 1}
                            except json.JSONDecodeError:
                                continue
                                
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
            
            # 保存到缓存
            self._save_to_cache(context_length, (queries, corpus, qrels))

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True 