#!/usr/bin/env python3
"""
OpenAI Long Embed评估脚本 - 使用本地修改的类
支持65536长度的数据评估
"""

import os
import json
import logging
import sys
import numpy as np
import shutil

# 确保能导入本地的类
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mteb import MTEB
from utils import logger, get_args
from encoder_model import RetrievalModel
from openai_model import OpenAIEmbeddingModel

# 直接导入本地修改的类
from LEMBNeedleRetrieval import LEMBNeedleRetrieval
from LEMBPasskeyRetrieval import LEMBPasskeyRetrieval

logging.getLogger().setLevel(logging.INFO)


class LocalTaskEvaluator:
    """使用本地任务类的评估器"""

    def __init__(self, model, output_dir, batch_size):
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size

    def evaluate_local_tasks(self, task_names, context_length_list):
        """评估本地任务类"""
        results = {}

        for task_name in task_names:
            logger.info(f"开始评估 {task_name}...")
            task_results = {}

            # 对每个context_length分别进行评估
            for ctx_len in context_length_list:
                logger.info(f"  评估 {task_name} context_length={ctx_len}")

                try:
                    # 为每个context_length创建任务实例
                    if task_name == "LEMBNeedleRetrieval":
                        task = LEMBNeedleRetrieval()
                    elif task_name == "LEMBPasskeyRetrieval":
                        task = LEMBPasskeyRetrieval()
                    else:
                        logger.warning(f"未知的任务: {task_name}")
                        continue

                    # 预先加载指定长度的数据
                    task.load_data(context_length=ctx_len)

                    # 检查是否有数据
                    if not task.queries or not task.corpus or not task.relevant_docs:
                        logger.warning(f"  跳过 {task_name} context_length={ctx_len} - 无数据")
                        continue

                    # 创建带缓存功能的模型包装器
                    cached_model = task.create_cached_model_wrapper(self.model)
                    logger.info(f"  🚀 已启用编码缓存: {task_name} context_length={ctx_len}")
                    
                    # 验证缓存包装器的pool_type传递
                    if hasattr(cached_model, 'pool_type') and hasattr(self.model, 'pool_type'):
                        if cached_model.pool_type == self.model.pool_type:
                            logger.info(f"  ✅ 缓存包装器pool_type正确传递: {cached_model.pool_type}")
                        else:
                            logger.warning(f"  ⚠️ 缓存包装器pool_type不一致: wrapper={cached_model.pool_type}, original={self.model.pool_type}")
                    elif hasattr(self.model, 'pool_type'):
                        logger.warning(f"  ⚠️ 缓存包装器缺少pool_type属性，原模型有: {self.model.pool_type}")
                    else:
                        logger.info(f"  📝 原模型和缓存包装器都没有pool_type属性")

                    # 为每个长度创建独立的输出目录
                    length_output_dir = os.path.join(self.output_dir, f"ctx_{ctx_len}")
                    os.makedirs(length_output_dir, exist_ok=True)

                    # 创建一个包含单个任务的MTEB评估器
                    evaluation = MTEB(tasks=[task])
                    
                    # 运行评估（使用带缓存的模型和独立输出目录）
                    single_results = evaluation.run(
                        cached_model,  # 使用缓存包装器而不是原始模型
                        output_folder=length_output_dir,  # 使用独立的输出目录
                        overwrite_results=True,
                        batch_size=self.batch_size,
                        verbosity=0,
                    )
                    
                    # 提取该context_length的结果（使用原始任务名称）
                    logger.info(f"  🔍 调试信息: 单次评估结果键 = {list(single_results.keys())}")
                    
                    if task_name in single_results:
                        logger.info(f"  🔍 找到任务 {task_name}，可用分片: {list(single_results[task_name].keys())}")
                        
                        # 通常会有test split的结果
                        test_split_key = f"test_{ctx_len}" if f"test_{ctx_len}" in single_results[task_name] else "test"
                        if test_split_key in single_results[task_name] or "test" in single_results[task_name]:
                            actual_key = test_split_key if test_split_key in single_results[task_name] else "test"
                            result_data = single_results[task_name][actual_key]
                            
                            logger.info(f"  🔍 提取结果数据 {actual_key}: {result_data}")
                            
                            task_results[f"test_{ctx_len}"] = result_data
                            logger.info(f"  ✓ {task_name} context_length={ctx_len} 完成")
                            
                            # 将生成的文件复制到主输出目录，并重命名
                            # self._copy_and_rename_result_file(
                            #     task_name, ctx_len, length_output_dir, self.output_dir
                            # )
                        else:
                            logger.warning(f"  ❌ {task_name} context_length={ctx_len} 未找到测试结果")
                            logger.warning(f"  🔍 期望的键: {test_split_key}，实际的键: {list(single_results[task_name].keys())}")
                    else:
                        logger.warning(f"  ❌ {task_name} context_length={ctx_len} 评估未返回结果")
                        logger.warning(f"  🔍 期望的任务名: {task_name}，实际的任务名: {list(single_results.keys())}")
                        
                        # 尝试使用实际存在的任务名（可能带有长度信息）
                        actual_task_names = [k for k in single_results.keys() if task_name in k]
                        if actual_task_names:
                            actual_task_name = actual_task_names[0]
                            logger.info(f"  🔧 尝试使用实际任务名: {actual_task_name}")
                            
                            if "test" in single_results[actual_task_name]:
                                result_data = single_results[actual_task_name]["test"]
                                task_results[f"test_{ctx_len}"] = result_data
                                logger.info(f"  ✓ {task_name} context_length={ctx_len} 完成（使用实际任务名）")
                                
                                # 将生成的文件复制到主输出目录，并重命名
                                # self._copy_and_rename_result_file(
                                #     actual_task_name, ctx_len, length_output_dir, self.output_dir
                                # )

                except Exception as e:
                    logger.error(f"  ❌ {task_name} context_length={ctx_len} 失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            if task_results:
                results[task_name] = task_results
                logger.info(f"✓ {task_name} 所有长度评估完成")

        return results

    def _copy_and_rename_result_file(self, task_name, ctx_len, source_dir, target_dir):
        """
        将特定context_length的结果文件从source_dir复制到target_dir，并重命名。
        文件名格式：LEMBNeedleRetrieval_1024.json
        """
        try:
            # MTEB会生成名为 {task_name}.json 的文件
            source_file = f"{task_name}.json"
            source_path = os.path.join(source_dir, source_file)
            
            # 目标文件名包含context_length
            target_file = f"{task_name}_{ctx_len}.json"
            target_path = os.path.join(target_dir, target_file)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                logger.info(f"  📁 结果文件已保存: {target_file}")
            else:
                logger.warning(f"  ❌ 源文件未找到: {source_path}")
                
        except Exception as e:
            logger.error(f"  ❌ 复制文件失败: {e}")


def main():
    args = get_args()

    # Choose model based on whether using OpenAI
    if args.use_openai:
        if not args.api_key or not args.base_url or not args.model_name:
            logger.error(
                "When using OpenAI mode, api_key, base_url and model_name parameters must be provided"
            )
            exit(1)

        logger.info("Using OpenAI embedding model")
        model = OpenAIEmbeddingModel(args)
        model_name = args.model_name
    else:
        if not args.model_name_or_path:
            logger.error(
                "When using local mode, model_name_or_path parameter must be provided"
            )
            exit(1)

        logger.info("Using local embedding model")
        model = RetrievalModel(args)
        model_name = os.path.basename(os.path.normpath(args.model_name_or_path))

    # 验证pool_type设置
    if hasattr(model, 'pool_type'):
        logger.info(f"✅ 模型pool_type已设置: {model.pool_type}")
    else:
        logger.warning("⚠️ 模型缺少pool_type属性")
    
    # 打印模型关键配置（用于调试缓存）
    logger.info("🔧 模型配置用于缓存键计算:")
    logger.info(f"  - model_name: {getattr(model, 'model_name', 'N/A')}")
    logger.info(f"  - pool_type: {getattr(model, 'pool_type', 'N/A')}")
    logger.info(f"  - prefix_type: {getattr(model, 'prefix_type', 'N/A')}")
    logger.info(f"  - l2_norm: {getattr(model, 'l2_norm', 'N/A')}")
    if hasattr(model, 'encode_max_length'):
        logger.info(f"  - encode_max_length: {model.encode_max_length}")
    if hasattr(model, 'model_name_or_path'):
        logger.info(f"  - model_name_or_path: {model.model_name_or_path}")

    # Set output directory
    mteb_output_dir = os.path.join(args.output_dir, model_name)

    # Handle chunking mode
    chunking_mode: str = os.getenv("CHUNKING_MODE")
    if chunking_mode != "no_chunk":
        chunk_max_len = os.getenv("MAX_TOKEN_NUM", "0")
        mteb_output_dir += f"_{chunking_mode}-{chunk_max_len}"

    # Handle position mode (only applies to local models)
    if not args.use_openai:
        if args.pos_mode != "original":
            mteb_output_dir += f"_{args.pos_mode}"
        if args.use_self_extend:
            mteb_output_dir += f"_se_{model.encode_max_length}"
        if args.rope_theta != 10000:
            mteb_output_dir += f"_theta{args.rope_theta}_{model.encode_max_length}"
        if args.rotary_scaling_factor != None:
            mteb_output_dir += f"_rsf{args.rotary_scaling_factor}"

    # If using OpenAI, add identifier
    if args.use_openai:
        mteb_output_dir += "_openai_local"
        # 添加pool_type到目录名（仅当使用OpenAI且有pool_type时）
        if hasattr(model, 'pool_type') and model.pool_type != 'avg':  # avg是默认值，不加到目录名
            mteb_output_dir += f"_pool_{model.pool_type}"
    else:
        mteb_output_dir += "_local"
        # 对于本地模型，如果pool_type不是模型默认值，也加到目录名
        if hasattr(model, 'pool_type') and model.pool_type != 'avg':
            mteb_output_dir += f"_pool_{model.pool_type}"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(mteb_output_dir, exist_ok=True)

    # 准备任务列表
    needle_passkey_task_list = []
    for task in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        if task in args.task_list:
            needle_passkey_task_list.append(task)

    output_dict = {}

    # 使用本地任务评估器
    if needle_passkey_task_list:
        logger.info(f"开始评估本地needle和passkey任务: {needle_passkey_task_list}")

        context_length_list = list(args.window_length_list)
        context_length_list.sort()

        evaluator = LocalTaskEvaluator(model, mteb_output_dir, args.batch_size)
        
        # 添加缓存管理功能
        if needle_passkey_task_list:
            # 显示缓存统计信息
            logger.info("📊 缓存统计信息:")
            for task_name in needle_passkey_task_list:
                if task_name == "LEMBNeedleRetrieval":
                    task_instance = LEMBNeedleRetrieval()
                elif task_name == "LEMBPasskeyRetrieval":
                    task_instance = LEMBPasskeyRetrieval()
                else:
                    continue
                
                cache_info = task_instance.get_cache_info()
                logger.info(f"  {task_name}: {cache_info['data_cache_files']} 数据缓存, {cache_info['encoding_cache_files']} 编码缓存, {cache_info['total_cache_size_mb']:.1f} MB")
                
                # 如果用户设置了环境变量来清理缓存
                clear_cache_days = os.getenv("CLEAR_CACHE_DAYS")
                if clear_cache_days:
                    try:
                        days = int(clear_cache_days)
                        logger.info(f"🧹 清理 {days} 天前的编码缓存...")
                        task_instance.clear_encoding_cache(older_than_days=days)
                    except ValueError:
                        logger.warning(f"无效的 CLEAR_CACHE_DAYS 值: {clear_cache_days}")
        
        results = evaluator.evaluate_local_tasks(needle_passkey_task_list, context_length_list)

        # 处理结果格式（为了兼容overall_results.json）
        for key, value in results.items():
            needle_passkey_score_list = []
            
            # 收集所有长度的结果
            for ctx_len in context_length_list:
                if f"test_{ctx_len}" in value:
                    score = value[f"test_{ctx_len}"]["ndcg_at_1"]
                    needle_passkey_score_list.append([ctx_len, score])
                    logger.info(f"  收集结果: {key} context_length={ctx_len}, ndcg@1={score:.4f}")

            if needle_passkey_score_list:
                # 计算平均值
                avg_score = sum([x[1] for x in needle_passkey_score_list]) / len(needle_passkey_score_list)
                needle_passkey_score_list.append(["avg", avg_score])
                
                # 转换为字典格式
                output_dict[key] = {
                    item[0]: item[1] for item in needle_passkey_score_list
                }
                logger.info(f"  ✓ {key} 所有长度结果已汇聚，平均分: {avg_score:.4f}")
            else:
                logger.warning(f"  ❌ {key} 没有收集到任何结果")

    # 处理其他检索任务（如果需要）
    retrieval_task_list = []
    for task in [
        "LEMBSummScreenFDRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBWikimQARetrieval",
        "LEMBNarrativeQARetrieval",
    ]:
        if task in args.task_list:
            retrieval_task_list.append(task)

    if retrieval_task_list:
        logger.info(f"Start evaluating retrieval tasks: {retrieval_task_list}")

        evaluation = MTEB(tasks=retrieval_task_list)
        results = evaluation.run(
            model,
            output_folder=mteb_output_dir,
            overwrite_results=False,
            batch_size=args.batch_size,
            verbosity=0,
        )

        for key, value in results.items():
            split = "test" if "test" in value else "validation"
            output_dict[key] = {
                "ndcg@1": value[split]["ndcg_at_1"],
                "ndcg@10": value[split]["ndcg_at_10"],
            }

    logger.info("Evaluation results:")
    logger.info(json.dumps(output_dict, indent=2, ensure_ascii=False))

    # Save results
    results_file = os.path.join(mteb_output_dir, "overall_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
