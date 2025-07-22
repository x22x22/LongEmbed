import datasets
import os
import json
import pickle
import hashlib
import time
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBPasskeyRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    _CACHE_DIR = "./cache/passkey"

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

    def _get_cache_path(self, context_length):
        """获取缓存文件路径"""
        os.makedirs(self._CACHE_DIR, exist_ok=True)
        return os.path.join(self._CACHE_DIR, f"passkey_data_{context_length}.pkl")

    def _get_file_hash(self, file_path):
        """获取文件的哈希值，用于检测文件是否更改"""
        if not os.path.exists(file_path):
            return None
        
        # 为了提高效率，只计算文件的部分哈希（前1MB+文件大小+修改时间）
        file_size = os.path.getsize(file_path)
        file_mtime = os.path.getmtime(file_path)
        
        hash_obj = hashlib.md5()
        hash_obj.update(str(file_size).encode())
        hash_obj.update(str(file_mtime).encode())
        
        # 读取文件前1MB用于哈希
        with open(file_path, 'rb') as f:
            chunk = f.read(1024 * 1024)  # 1MB
            hash_obj.update(chunk)
            
        return hash_obj.hexdigest()

    def _load_from_cache(self, context_length):
        """从缓存加载数据"""
        cache_path = self._get_cache_path(context_length)
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            # 检查缓存版本和文件哈希
            local_data_dir = "./local_data/passkey"
            if os.path.exists(f"{local_data_dir}/corpus.jsonl"):
                current_hash = self._get_file_hash(f"{local_data_dir}/corpus.jsonl")
                if cached_data.get('file_hash') != current_hash:
                    print("检测到数据文件已更改，缓存失效")
                    return None
                    
            print(f"✅ 从缓存快速加载数据: context_length={context_length}")
            return cached_data['data']
            
        except Exception as e:
            print(f"⚠️  缓存加载失败: {e}")
            return None

    def _save_to_cache(self, context_length, data):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(context_length)
        
        # 获取文件哈希用于版本控制
        local_data_dir = "./local_data/passkey"
        file_hash = None
        if os.path.exists(f"{local_data_dir}/corpus.jsonl"):
            file_hash = self._get_file_hash(f"{local_data_dir}/corpus.jsonl")
            
        cached_data = {
            'data': data,
            'file_hash': file_hash,
            'context_length': context_length,
            'cache_time': time.time()
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 数据已缓存: {cache_path}")
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

    def _stream_filter_jsonl(self, file_path, context_length, file_type):
        """流式读取JSONL文件并过滤指定长度的数据"""
        result = {}
        total_lines = 0
        matched_lines = 0
        
        print(f"🔄 [{file_type}] 流式处理 {os.path.basename(file_path)}...")
        start_time = time.time()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                if total_lines % 20000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   [{file_type}] 已处理 {total_lines:,} 行，匹配 {matched_lines:,} 行 ({elapsed:.1f}s)")
                
                try:
                    data = json.loads(line.strip())
                    if data.get('context_length') == context_length:
                        if file_type == "queries":
                            result[data["qid"]] = data["text"]
                        elif file_type == "corpus":
                            result[data["doc_id"]] = {"text": data["text"]}
                        elif file_type == "qrels":
                            if data["qid"] not in result:
                                result[data["qid"]] = {}
                            result[data["qid"]][data["doc_id"]] = 1
                        matched_lines += 1
                except json.JSONDecodeError:
                    continue
                    
        elapsed = time.time() - start_time
        print(f"✅ [{file_type}] 完成：总共 {total_lines:,} 行，匹配 {matched_lines:,} 行 ({elapsed:.1f}s)")
        return result

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        print(f"🚀 开始加载passkey数据，context_length={context_length}")
        start_time = time.time()

        # 尝试从缓存加载
        cached_data = self._load_from_cache(context_length)
        if cached_data:
            queries, corpus, qrels = cached_data
            elapsed = time.time() - start_time
            print(f"🎉 缓存加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系 ({elapsed:.1f}s)")
        else:
            # 缓存未命中，使用流式处理加载数据
            local_data_dir = "./local_data/passkey"
            use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                             os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                             os.path.exists(f"{local_data_dir}/qrels.jsonl"))

            if use_local_data:
                print(f"💽 使用本地数据流式加载passkey数据，context_length={context_length}")
                
                # 流式处理每个文件
                queries = self._stream_filter_jsonl(f"{local_data_dir}/queries.jsonl", context_length, "queries")
                corpus = self._stream_filter_jsonl(f"{local_data_dir}/corpus.jsonl", context_length, "corpus")
                qrels = self._stream_filter_jsonl(f"{local_data_dir}/qrels.jsonl", context_length, "qrels")
                
            else:
                print(f"🌐 使用HuggingFace数据加载passkey数据，context_length={context_length}")
                # 从HuggingFace加载（保持原有逻辑作为后备）
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
                qrels = {}
                for row in qrels_list:
                    if row["qid"] not in qrels:
                        qrels[row["qid"]] = {}
                    qrels[row["qid"]][row["doc_id"]] = 1

            elapsed = time.time() - start_time
            print(f"✅ 流式加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系 ({elapsed:.1f}s)")
            
            # 保存到缓存（在后台异步进行，避免阻塞）
            try:
                self._save_to_cache(context_length, (queries, corpus, qrels))
            except Exception as e:
                print(f"⚠️  缓存保存异常: {e}")

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True 