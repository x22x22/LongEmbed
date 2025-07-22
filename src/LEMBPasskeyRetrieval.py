import datasets
import os
import json
import pickle
import hashlib
import time
import numpy as np
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBPasskeyRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    _CACHE_DIR = "./cache/passkey"
    _ENCODING_CACHE_DIR = "./cache/passkey/encodings"

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

    def _get_encoding_cache_path(self, cache_key, encoding_type="default"):
        """获取编码缓存文件路径"""
        os.makedirs(self._ENCODING_CACHE_DIR, exist_ok=True)
        return os.path.join(self._ENCODING_CACHE_DIR, f"encoding_{encoding_type}_{cache_key}.npy")

    def _get_comprehensive_cache_key(self, model, texts, encoding_type="default", batch_size=32, **kwargs):
        """获取综合的缓存键，包含文本、模型、参数等信息"""
        # 1. 文本内容哈希
        if isinstance(texts, str):
            texts = [texts]
        
        if isinstance(texts, list):
            # 对于文本列表，计算整个列表的哈希
            combined_text = "\n".join(str(t) for t in texts)
        else:
            combined_text = str(texts)
        
        # 2. 构建完整的缓存键组件
        cache_components = [
            f"text_hash:{hashlib.md5(combined_text.encode('utf-8')).hexdigest()}",
            f"encoding_type:{encoding_type}",
            f"batch_size:{batch_size}",
            f"text_count:{len(texts)}"
        ]
        
        # 3. 添加模型标识符
        try:
            # 尝试获取模型的唯一标识
            if hasattr(model, 'model_name_or_path'):
                model_id = str(model.model_name_or_path)
            elif hasattr(model, 'model_name'):
                model_id = str(model.model_name)
            elif hasattr(model, '__class__'):
                model_id = model.__class__.__name__
            else:
                model_id = str(type(model))
            
            cache_components.append(f"model:{hashlib.md5(model_id.encode('utf-8')).hexdigest()[:8]}")
        except:
            cache_components.append("model:unknown")
        
        # 4. 添加pool_type（重要！）
        pool_type = None
        # 优先从kwargs中获取pool_type
        if 'pool_type' in kwargs:
            pool_type = kwargs['pool_type']
        # 其次从模型属性中获取
        elif hasattr(model, 'pool_type'):
            pool_type = model.pool_type
        # 如果找不到，使用默认值
        if pool_type is None:
            pool_type = "unknown"
        
        cache_components.append(f"pool_type:{pool_type}")
        
        # 5. 添加其他重要的kwargs参数
        if kwargs:
            # 只包含可能影响编码结果的重要参数（排除pool_type，因为已经单独处理）
            important_params = {}
            for key, value in kwargs.items():
                if key in ['max_length', 'truncation', 'padding', 'return_tensors', 'normalize_embeddings', 
                          'pos_mode', 'encode_max_length', 'prefix_type', 'l2_norm']:
                    important_params[key] = value
            
            if important_params:
                params_str = json.dumps(important_params, sort_keys=True, ensure_ascii=False)
                params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()[:8]
                cache_components.append(f"params:{params_hash}")
        
        # 6. 添加模型的其他关键属性
        try:
            model_attrs = []
            if hasattr(model, 'prefix_type'):
                model_attrs.append(f"prefix_type:{model.prefix_type}")
            if hasattr(model, 'l2_norm'):
                model_attrs.append(f"l2_norm:{model.l2_norm}")
            if hasattr(model, 'encode_max_length'):
                model_attrs.append(f"max_len:{model.encode_max_length}")
            
            if model_attrs:
                model_attrs_str = "|".join(model_attrs)
                model_attrs_hash = hashlib.md5(model_attrs_str.encode('utf-8')).hexdigest()[:8]
                cache_components.append(f"model_attrs:{model_attrs_hash}")
        except:
            pass
        
        # 7. 生成最终的缓存键
        cache_key_str = "|".join(cache_components)
        final_cache_key = hashlib.md5(cache_key_str.encode('utf-8')).hexdigest()
        
        return final_cache_key

    def _load_encoding_from_cache(self, cache_key, encoding_type="default"):
        """从缓存加载编码结果"""
        cache_path = self._get_encoding_cache_path(cache_key, encoding_type)
        if os.path.exists(cache_path):
            try:
                encoding = np.load(cache_path)
                return encoding
            except Exception as e:
                print(f"⚠️ 编码缓存加载失败: {e}")
                return None
        return None

    def _save_encoding_to_cache(self, cache_key, encoding, encoding_type="default"):
        """保存编码结果到缓存"""
        cache_path = self._get_encoding_cache_path(cache_key, encoding_type)
        try:
            np.save(cache_path, encoding)
        except Exception as e:
            print(f"⚠️ 编码缓存保存失败: {e}")

    def encode_with_cache(self, model, texts, batch_size=32, encoding_type="default", **kwargs):
        """带缓存的编码方法"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 生成综合缓存键
        cache_key = self._get_comprehensive_cache_key(model, texts, encoding_type, batch_size, **kwargs)
        
        # 收集调试信息
        debug_info = []
        debug_info.append(f"texts={len(texts)}")
        debug_info.append(f"type={encoding_type}")
        debug_info.append(f"batch={batch_size}")
        
        # 添加pool_type调试信息
        pool_type = None
        if 'pool_type' in kwargs:
            pool_type = kwargs['pool_type']
        elif hasattr(model, 'pool_type'):
            pool_type = model.pool_type
        if pool_type:
            debug_info.append(f"pool={pool_type}")
        
        # 添加模型信息
        if hasattr(model, 'model_name_or_path'):
            model_short = os.path.basename(str(model.model_name_or_path))
            debug_info.append(f"model={model_short}")
        elif hasattr(model, '__class__'):
            debug_info.append(f"model={model.__class__.__name__}")
        
        debug_str = ", ".join(debug_info)
        
        # 尝试从缓存加载
        cached_encoding = self._load_encoding_from_cache(cache_key, encoding_type)
        if cached_encoding is not None:
            print(f"🎯 编码缓存命中: {debug_str} (key: {cache_key[:8]}...)")
            return cached_encoding
        
        # 缓存未命中，调用模型编码
        print(f"🔄 编码缓存未命中，调用模型编码: {debug_str} (key: {cache_key[:8]}...)")
        start_time = time.time()
        
        # 根据编码类型选择不同的编码方法
        if encoding_type == "queries" and hasattr(model, "encode_queries"):
            encoding = model.encode_queries(texts, batch_size=batch_size, **kwargs)
        elif encoding_type == "corpus" and hasattr(model, "encode_corpus"):
            # 转换为corpus格式
            corpus_format = [{"text": text} for text in texts]
            encoding = model.encode_corpus(corpus_format, batch_size=batch_size, **kwargs)
        else:
            # 默认编码方法
            encoding = model.encode(texts, batch_size=batch_size, **kwargs)
        
        elapsed = time.time() - start_time
        print(f"✅ 模型编码完成: {debug_str}, 耗时 {elapsed:.1f}s")
        
        # 保存到缓存
        self._save_encoding_to_cache(cache_key, encoding, encoding_type)
        
        return encoding

    def create_cached_model_wrapper(self, original_model):
        """创建带缓存功能的模型包装器"""
        class CachedModelWrapper:
            def __init__(self, model, task_instance):
                self.model = model
                self.task = task_instance
                # 复制原模型的所有其他属性
                for attr in dir(model):
                    if not attr.startswith('_') and attr not in ['encode', 'encode_queries', 'encode_corpus']:
                        try:
                            setattr(self, attr, getattr(model, attr))
                        except:
                            pass
            
            def encode(self, texts, batch_size=32, **kwargs):
                return self.task.encode_with_cache(
                    self.model, texts, batch_size=batch_size, encoding_type="default", **kwargs
                )
            
            def encode_queries(self, texts, batch_size=32, **kwargs):
                return self.task.encode_with_cache(
                    self.model, texts, batch_size=batch_size, encoding_type="queries", **kwargs
                )
            
            def encode_corpus(self, corpus, batch_size=32, **kwargs):
                # 提取文本用于缓存键计算
                if isinstance(corpus, list) and len(corpus) > 0:
                    if isinstance(corpus[0], dict) and 'text' in corpus[0]:
                        texts = [item['text'] for item in corpus]
                    else:
                        texts = corpus
                else:
                    texts = corpus
                
                return self.task.encode_with_cache(
                    self.model, texts, batch_size=batch_size, encoding_type="corpus", **kwargs
                )
        
        return CachedModelWrapper(original_model, self)

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

    def clear_encoding_cache(self, older_than_days=7):
        """清理编码缓存目录中的旧文件"""
        if not os.path.exists(self._ENCODING_CACHE_DIR):
            print("📁 编码缓存目录不存在，无需清理")
            return
        
        import time
        current_time = time.time()
        days_in_seconds = older_than_days * 24 * 60 * 60
        
        cleared_count = 0
        total_count = 0
        
        print(f"🧹 开始清理 {older_than_days} 天前的编码缓存...")
        
        for filename in os.listdir(self._ENCODING_CACHE_DIR):
            if filename.endswith('.npy'):
                file_path = os.path.join(self._ENCODING_CACHE_DIR, filename)
                total_count += 1
                
                try:
                    file_mtime = os.path.getmtime(file_path)
                    if current_time - file_mtime > days_in_seconds:
                        os.remove(file_path)
                        cleared_count += 1
                except Exception as e:
                    print(f"⚠️ 清理文件失败 {filename}: {e}")
        
        print(f"✅ 编码缓存清理完成: 清理了 {cleared_count}/{total_count} 个文件")

    def get_cache_info(self):
        """获取缓存信息统计"""
        info = {
            "data_cache_dir": self._CACHE_DIR,
            "encoding_cache_dir": self._ENCODING_CACHE_DIR,
            "data_cache_files": 0,
            "encoding_cache_files": 0,
            "total_cache_size_mb": 0
        }
        
        # 统计数据缓存
        if os.path.exists(self._CACHE_DIR):
            for filename in os.listdir(self._CACHE_DIR):
                if filename.endswith('.pkl'):
                    info["data_cache_files"] += 1
                    file_path = os.path.join(self._CACHE_DIR, filename)
                    info["total_cache_size_mb"] += os.path.getsize(file_path) / (1024 * 1024)
        
        # 统计编码缓存
        if os.path.exists(self._ENCODING_CACHE_DIR):
            for filename in os.listdir(self._ENCODING_CACHE_DIR):
                if filename.endswith('.npy'):
                    info["encoding_cache_files"] += 1
                    file_path = os.path.join(self._ENCODING_CACHE_DIR, filename)
                    info["total_cache_size_mb"] += os.path.getsize(file_path) / (1024 * 1024)
        
        return info 