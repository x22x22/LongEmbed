import datasets
import os
import json
import sqlite3
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBNeedleRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    _DB_DIR = "./database"

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

    def _get_db_path(self):
        """获取数据库文件路径"""
        os.makedirs(self._DB_DIR, exist_ok=True)
        return os.path.join(self._DB_DIR, "needle_data.db")

    def _init_database(self):
        """初始化数据库表结构"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                qid TEXT PRIMARY KEY,
                text TEXT,
                context_length INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corpus (
                doc_id TEXT PRIMARY KEY,
                text TEXT,
                context_length INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qrels (
                qid TEXT,
                doc_id TEXT,
                context_length INTEGER,
                PRIMARY KEY (qid, doc_id)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_queries_context_length ON queries(context_length)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_corpus_context_length ON corpus(context_length)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_qrels_context_length ON qrels(context_length)')
        
        conn.commit()
        conn.close()

    def _is_database_populated(self):
        """检查数据库是否已有数据"""
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return False
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM queries')
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except sqlite3.OperationalError:
            conn.close()
            return False

    def _populate_database(self):
        """将JSONL数据导入SQLite数据库"""
        local_data_dir = "./local_data/needle"
        
        if not all(os.path.exists(f"{local_data_dir}/{file}.jsonl") 
                  for file in ["queries", "corpus", "qrels"]):
            print("本地数据文件不存在，跳过数据库填充")
            return False
            
        print("正在将数据导入SQLite数据库...")
        
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 清空现有数据
        cursor.execute('DELETE FROM queries')
        cursor.execute('DELETE FROM corpus')
        cursor.execute('DELETE FROM qrels')
        
        # 导入queries
        print("导入queries数据...")
        with open(f"{local_data_dir}/queries.jsonl", 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    cursor.execute(
                        'INSERT OR REPLACE INTO queries (qid, text, context_length) VALUES (?, ?, ?)',
                        (data['qid'], data['text'], data['context_length'])
                    )
                    if line_num % 1000 == 0:
                        conn.commit()
                        print(f"  已导入 {line_num} 条queries记录")
                except json.JSONDecodeError:
                    continue
        
        # 导入corpus
        print("导入corpus数据...")
        with open(f"{local_data_dir}/corpus.jsonl", 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    cursor.execute(
                        'INSERT OR REPLACE INTO corpus (doc_id, text, context_length) VALUES (?, ?, ?)',
                        (data['doc_id'], data['text'], data['context_length'])
                    )
                    if line_num % 1000 == 0:
                        conn.commit()
                        print(f"  已导入 {line_num} 条corpus记录")
                except json.JSONDecodeError:
                    continue
        
        # 导入qrels
        print("导入qrels数据...")
        with open(f"{local_data_dir}/qrels.jsonl", 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    cursor.execute(
                        'INSERT OR REPLACE INTO qrels (qid, doc_id, context_length) VALUES (?, ?, ?)',
                        (data['qid'], data['doc_id'], data['context_length'])
                    )
                    if line_num % 1000 == 0:
                        conn.commit()
                        print(f"  已导入 {line_num} 条qrels记录")
                except json.JSONDecodeError:
                    continue
        
        conn.commit()
        conn.close()
        print("数据库导入完成!")
        return True

    def _load_from_database(self, context_length):
        """从SQLite数据库加载指定长度的数据"""
        db_path = self._get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"从数据库查询 context_length={context_length} 的数据...")
        
        # 查询queries
        cursor.execute('SELECT qid, text FROM queries WHERE context_length = ?', (context_length,))
        queries = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 查询corpus
        cursor.execute('SELECT doc_id, text FROM corpus WHERE context_length = ?', (context_length,))
        corpus = {row[0]: {"text": row[1]} for row in cursor.fetchall()}
        
        # 查询qrels
        cursor.execute('SELECT qid, doc_id FROM qrels WHERE context_length = ?', (context_length,))
        qrels = {}
        for qid, doc_id in cursor.fetchall():
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = 1
        
        conn.close()
        return queries, corpus, qrels

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        if "context_length" not in kwargs:
            raise ValueError("Need to specify context_length")
        context_length = kwargs["context_length"]

        # 初始化数据库
        self._init_database()
        
        # 检查数据库是否需要填充
        if not self._is_database_populated():
            if not self._populate_database():
                # 数据库填充失败，回退到原始方法
                print("数据库填充失败，使用原始加载方法")
                local_data_dir = "./local_data/needle"
                use_local_data = (os.path.exists(f"{local_data_dir}/queries.jsonl") and
                                 os.path.exists(f"{local_data_dir}/corpus.jsonl") and
                                 os.path.exists(f"{local_data_dir}/qrels.jsonl"))

                if use_local_data:
                    print(f"使用本地数据加载needle数据，context_length={context_length}")
                    # 使用原始的datasets.Dataset.from_json方法
                    query_list = datasets.Dataset.from_json(f"{local_data_dir}/queries.jsonl")
                    corpus_list = datasets.Dataset.from_json(f"{local_data_dir}/corpus.jsonl")
                    qrels_list = datasets.Dataset.from_json(f"{local_data_dir}/qrels.jsonl")
                    
                    # 过滤指定长度的数据
                    query_list = query_list.filter(lambda x: x["context_length"] == context_length)
                    queries = {row["qid"]: row["text"] for row in query_list}

                    corpus_list = corpus_list.filter(
                        lambda x: x["context_length"] == context_length
                    )
                    corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

                    qrels_list = qrels_list.filter(lambda x: x["context_length"] == context_length)
                    qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}
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
        else:
            # 从数据库加载
            queries, corpus, qrels = self._load_from_database(context_length)

        print(f"数据库加载完成：{len(queries)} 个查询，{len(corpus)} 个文档，{len(qrels)} 个关系")

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True 