#!/usr/bin/env python3
"""
为LongEmbed生成扩展长度的数据集
方案：保持查询(s)不变，将两个不同长度的文档(p)合并成目标长度的文档
支持句子级别的随机打乱功能
"""

from datasets import load_dataset, Dataset
import json
import os
import random
import argparse
from typing import Dict, List, Any

def shuffle_sentences(text: str, seed: int = 42) -> str:
    """
    按照'.\n'分割文本，打乱句子顺序后重新拼接
    
    Args:
        text: 输入文本
        seed: 随机种子
    
    Returns:
        打乱后的文本
    """
    # 按照".\n"分割成句子列表
    sentences = text.split('.\n')
    
    # 保存最后一个句子的结尾（如果不是以'.\n'结尾）
    last_sentence = sentences[-1]
    sentences = sentences[:-1]  # 移除最后一个可能不完整的句子
    
    # 打乱句子顺序
    random.seed(seed)
    random.shuffle(sentences)
    
    # 重新拼接，加上最后一个句子
    shuffled_text = '.\n'.join(sentences)
    if last_sentence.strip():  # 如果最后一个句子不为空
        shuffled_text += '.\n' + last_sentence
    
    return shuffled_text

def generate_extended_data(shuffle_sentences_flag: bool = False, shuffle_seed: int = 42, target_combinations: list = None):
    """为needle和passkey任务生成扩展长度的数据"""
    
    datasets_to_process = ["needle", "passkey"]
    
    # 默认的长度组合配置
    if target_combinations is None:
        target_combinations = [
            (32768, 32768, 65536),    # 原来的65536组合
            (32768, 256, 33024),      # 32768+256
            (32768, 512, 33280),      # 32768+512  
            (32768, 1024, 33792),     # 32768+1024
            (32768, 2048, 34816),     # 32768+2048
            (32768, 4096, 36864),     # 32768+4096
            (32768, 8192, 40960),     # 32768+8192
            (32768, 16384, 49152),    # 32768+16384
        ]
    
    print(f"将生成以下长度组合的数据: {[combo[2] for combo in target_combinations]}")
    
    # 加载所有需要的长度数据
    all_length_data = {}
    
    for dataset_name in datasets_to_process:
        print(f"\n=== 处理数据集: {dataset_name} ===")
        
        # 收集所有需要的长度
        required_lengths = set()
        for combo in target_combinations:
            required_lengths.update([combo[0], combo[1]])
        
        print(f"加载所需长度的数据: {sorted(required_lengths)}")
        
        # 加载原始数据
        full_queries = load_dataset("dwzhu/LongEmbed", name=dataset_name, split="queries")
        full_corpus = load_dataset("dwzhu/LongEmbed", name=dataset_name, split="corpus") 
        full_qrels = load_dataset("dwzhu/LongEmbed", name=dataset_name, split="qrels")
        
        # 按长度分组数据
        length_data = {}
        for length in required_lengths:
            queries_len = full_queries.filter(lambda x: x["context_length"] == length)
            corpus_len = full_corpus.filter(lambda x: x["context_length"] == length)
            qrels_len = full_qrels.filter(lambda x: x["context_length"] == length)
            
            length_data[length] = {
                'queries': queries_len,
                'corpus': corpus_len, 
                'qrels': qrels_len
            }
            print(f"  长度 {length}: {len(queries_len)} 查询, {len(corpus_len)} 文档, {len(qrels_len)} 关系")
        
        # 生成所有组合的数据
        all_new_queries = []
        all_new_corpus = []
        all_new_qrels = []
        
        for len1, len2, target_len in target_combinations:
            print(f"\n生成 {len1}+{len2}={target_len} 长度的数据...")
            
            if len(length_data[len1]['corpus']) == 0 or len(length_data[len2]['corpus']) == 0:
                print(f"跳过 {len1}+{len2} 组合 - 缺少必要的数据")
                continue
            
            new_queries, new_corpus, new_qrels = generate_s2p_combined_data(
                length_data[len1], length_data[len2], target_len, dataset_name, shuffle_sentences_flag, shuffle_seed
            )
            
            all_new_queries.extend(new_queries)
            all_new_corpus.extend(new_corpus)
            all_new_qrels.extend(new_qrels)
        
        # 合并原始数据和新数据
        print("合并原始数据和新生成的数据...")
        all_queries = combine_datasets(full_queries, Dataset.from_list(all_new_queries))
        all_corpus = combine_datasets(full_corpus, Dataset.from_list(all_new_corpus))
        all_qrels = combine_datasets(full_qrels, Dataset.from_list(all_new_qrels))
        
        # 保存到本地
        save_extended_data(dataset_name, all_queries, all_corpus, all_qrels)

def generate_s2p_combined_data(data1, data2, target_length, dataset_name, shuffle_sentences_flag=False, shuffle_seed=42):
    """生成s2p任务的组合长度数据"""
    
    print(f"生成{target_length}长度的s2p数据...")
    
    # 提取数据
    queries_len1 = data1['queries']
    corpus_len1 = data1['corpus']
    qrels_len1 = data1['qrels']
    
    corpus_len2 = data2['corpus']
    
    # 将文档按doc_id分组
    corpus_dict1 = {doc["doc_id"]: doc for doc in corpus_len1}
    corpus_dict2 = {doc["doc_id"]: doc for doc in corpus_len2}
    
    # 将查询按qid分组  
    queries_dict1 = {query["qid"]: query for query in queries_len1}
    
    # 将qrels按qid分组
    qrels_dict1 = {}
    for qrel in qrels_len1:
        qid = qrel["qid"]
        if qid not in qrels_dict1:
            qrels_dict1[qid] = []
        qrels_dict1[qid].append(qrel["doc_id"])
    
    # 生成新数据
    new_queries = []
    new_corpus = []
    new_qrels = []
    
    # 准备合并文档
    corpus_list1 = list(corpus_len1)
    corpus_list2 = list(corpus_len2)
    used_doc_pairs = set()
    
    # 为每个查询生成对应的目标长度文档
    for qid, original_doc_ids in qrels_dict1.items():
        if qid not in queries_dict1:
            continue
            
        original_query = queries_dict1[qid]
        
        # 创建新的查询（context_length改为目标长度，其他保持不变）
        new_query = {
            "qid": f"{qid}_{target_length}",
            "text": original_query["text"],  # 查询文本保持不变
            "context_length": target_length
        }
        new_queries.append(new_query)
        
        # 为这个查询创建合并的文档
        # 优先选择与原查询相关的文档作为第一个文档
        doc1 = None
        doc2 = None
        
        # 尝试找到与原查询相关的文档
        for doc_id in original_doc_ids:
            if doc_id in corpus_dict1:
                doc1 = corpus_dict1[doc_id]
                break
        
        # 如果没找到相关文档，随机选择第一个
        if doc1 is None and corpus_list1:
            doc1 = corpus_list1[0]
        
        # 随机选择第二个文档
        if corpus_list2:
            doc2 = random.choice(corpus_list2)
        
        if doc1 and doc2:
            # 合并两个文档
            first_doc_text = doc1["text"]
            second_doc_text = doc2["text"]
            
            # 根据开关决定是否打乱句子
            if shuffle_sentences_flag:
                print(f"    打乱文档句子顺序 (seed={shuffle_seed})")
                first_doc_text = shuffle_sentences(first_doc_text, shuffle_seed)
                second_doc_text = shuffle_sentences(second_doc_text, shuffle_seed + 1)  # 使用不同seed
            
            merged_text = first_doc_text + "\n\n--- Document Continuation ---\n\n" + second_doc_text
            
            new_doc_id = f"{doc1['doc_id']}_merged_{doc2['doc_id']}_{target_length}"
            new_doc = {
                "doc_id": new_doc_id,
                "text": merged_text,
                "context_length": target_length
            }
            new_corpus.append(new_doc)
            
            # 记录使用过的文档对
            used_doc_pairs.add((doc1["doc_id"], doc2["doc_id"]))
            
            # 创建对应的qrel
            new_qrel = {
                "qid": new_query["qid"],
                "doc_id": new_doc_id,
                "text": "",  # qrels中的text字段应该是空字符串
                "context_length": target_length
            }
            new_qrels.append(new_qrel)
            
            print(f"  为查询 {qid} 创建了{target_length}长度的文档 {new_doc_id}")
    
    print(f"生成了 {len(new_queries)} 个查询，{len(new_corpus)} 个文档，{len(new_qrels)} 个关系")
    
    return new_queries, new_corpus, new_qrels

def combine_datasets(original_dataset, new_dataset):
    """合并原始数据集和新数据集"""
    
    # 转换为列表
    original_list = list(original_dataset)
    new_list = list(new_dataset)
    
    # 合并
    combined_list = original_list + new_list
    
    return Dataset.from_list(combined_list)

def save_extended_data(dataset_name, queries, corpus, qrels):
    """保存扩展后的数据到本地"""
    
    output_dir = f"./local_data/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到JSONL文件
    queries.to_json(f"{output_dir}/queries.jsonl")
    corpus.to_json(f"{output_dir}/corpus.jsonl")
    qrels.to_json(f"{output_dir}/qrels.jsonl")
    
    print(f"数据已保存到 {output_dir}/")
    
    # 统计各长度的数据量
    print("\n数据统计：")
    for split_name, dataset in [("queries", queries), ("corpus", corpus), ("qrels", qrels)]:
        lengths = {}
        for item in dataset:
            ctx_len = item.get("context_length", "unknown")
            lengths[ctx_len] = lengths.get(ctx_len, 0) + 1
        print(f"  {split_name}: {lengths}")

def verify_generated_data():
    """验证生成的数据"""
    
    print("\n=== 验证生成的数据 ===")
    
    datasets_to_check = ["needle", "passkey"]
    
    for dataset_name in datasets_to_check:
        print(f"\n验证数据集: {dataset_name}")
        
        data_dir = f"./local_data/{dataset_name}"
        
        if not os.path.exists(data_dir):
            print(f"  错误：目录 {data_dir} 不存在")
            continue
            
        for split in ["queries", "corpus", "qrels"]:
            file_path = f"{data_dir}/{split}.jsonl"
            if os.path.exists(file_path):
                dataset = Dataset.from_json(file_path)
                
                # 检查各种长度的数据
                length_counts = {}
                for item in dataset:
                    ctx_len = item.get("context_length", "unknown")
                    length_counts[ctx_len] = length_counts.get(ctx_len, 0) + 1
                
                print(f"  {split}: 总共 {len(dataset)} 条")
                for length, count in sorted(length_counts.items()):
                    print(f"    长度 {length}: {count} 条")
                
                # 抽样检查大于32768长度的数据质量
                for target_len in [33024, 33280, 33792, 34816, 36864, 40960, 49152, 65536]:
                    data_target = dataset.filter(lambda x: x.get("context_length") == target_len)
                    if len(data_target) > 0:
                        sample = data_target[0]
                        if split == "corpus":
                            text_length = len(sample["text"])
                            print(f"    样本文档长度{target_len}: {text_length} 字符")
                            if "Document Continuation" in sample["text"]:
                                print(f"      ✓ 文档包含合并标识")
                        elif split == "queries":
                            print(f"    样本查询{target_len}: {sample['text'][:50]}...")
            else:
                print(f"  错误：文件 {file_path} 不存在")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成扩展长度的LongEmbed数据集")
    parser.add_argument("--shuffle-sentences", action="store_true", 
                       help="是否在合并文档时打乱句子顺序（按'.\n'分割）")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                       help="句子打乱的随机种子 (默认: 42)")
    parser.add_argument("--combinations", type=str, nargs='*', 
                       help="自定义长度组合，格式：len1,len2,target (例如: 32768,256,33024)")
    
    args = parser.parse_args()
    
    print("开始生成扩展长度的LongEmbed数据集...")
    print("方案：保持查询不变，合并两个不同长度的文档创建目标长度的文档")
    
    # 解析自定义组合
    target_combinations = None
    if args.combinations:
        target_combinations = []
        for combo_str in args.combinations:
            len1, len2, target = map(int, combo_str.split(','))
            target_combinations.append((len1, len2, target))
        print(f"使用自定义组合: {target_combinations}")
    
    if args.shuffle_sentences:
        print(f"🔀 启用句子打乱功能 (seed={args.shuffle_seed})")
    else:
        print("📖 保持文档原始顺序")
    
    # 设置随机种子以保证可重复性
    random.seed(args.shuffle_seed)
    
    # 创建本地数据目录
    os.makedirs("local_data", exist_ok=True)
    
    # 生成数据
    generate_extended_data(shuffle_sentences_flag=args.shuffle_sentences, shuffle_seed=args.shuffle_seed, target_combinations=target_combinations)
    
    # 验证数据
    verify_generated_data()
    
    print("\n✓ 数据生成完成！")
    print("现在可以使用修改后的LEMBNeedleRetrieval.py和LEMBPasskeyRetrieval.py来加载扩展长度的数据。") 