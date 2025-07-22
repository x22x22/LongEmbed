#!/usr/bin/env python3
"""
pool_type 传递集成测试脚本
验证从命令行参数到缓存键的完整pool_type传递流程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import get_args
from openai_model import OpenAIEmbeddingModel
from encoder_model import RetrievalModel
from LEMBNeedleRetrieval import LEMBNeedleRetrieval

def test_pool_type_integration():
    """测试pool_type的完整传递流程"""
    
    print("🚀 开始pool_type集成测试")
    print("=" * 50)
    
    # 测试不同的pool_type值
    test_cases = [
        {"pool_type": "avg", "description": "平均池化"},
        {"pool_type": "cls", "description": "CLS池化"},
        {"pool_type": "last", "description": "最后token池化"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试案例 {i}: {case['description']} (pool_type={case['pool_type']})")
        print("-" * 40)
        
        # 模拟命令行参数
        test_args = [
            "--use_openai",
            "--api_key", "test-key",
            "--base_url", "http://localhost:31090/v1",
            "--model_name", "test-model",
            "--pool_type", case['pool_type'],
            "--output_dir", "/tmp/test",
            "--task_list", "LEMBNeedleRetrieval",
            "--window_length_list", "256"
        ]
        
        # 解析参数
        args = get_args(test_args)
        print(f"  ✅ 参数解析成功: pool_type = {args.pool_type}")
        
        # 创建OpenAI模型实例
        try:
            model = OpenAIEmbeddingModel(args)
            print(f"  ✅ OpenAI模型创建成功: pool_type = {model.pool_type}")
            
            # 验证pool_type设置
            assert model.pool_type == case['pool_type'], f"期望 {case['pool_type']}，实际 {model.pool_type}"
            print(f"  ✅ pool_type验证通过")
            
            # 测试缓存包装器
            task = LEMBNeedleRetrieval()
            cached_model = task.create_cached_model_wrapper(model)
            
            # 验证缓存包装器的pool_type
            if hasattr(cached_model, 'pool_type'):
                print(f"  ✅ 缓存包装器pool_type: {cached_model.pool_type}")
                assert cached_model.pool_type == case['pool_type'], f"缓存包装器pool_type不匹配"
                print(f"  ✅ 缓存包装器pool_type验证通过")
            else:
                print(f"  ❌ 缓存包装器缺少pool_type属性")
                return False
                
            # 测试缓存键生成
            test_texts = ["这是一个测试文本"]
            cache_key = task._get_comprehensive_cache_key(
                cached_model, test_texts, "queries", 32
            )
            print(f"  ✅ 缓存键生成成功: {cache_key[:16]}...")
            
            # 验证缓存键中包含pool_type信息
            # 这里我们无法直接验证，但可以通过生成不同pool_type的缓存键来确保它们不同
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 所有pool_type集成测试通过！")
    
    # 额外测试：验证不同pool_type生成不同缓存键
    print("\n🔍 额外测试：验证不同pool_type生成不同缓存键")
    print("-" * 40)
    
    task = LEMBNeedleRetrieval()
    test_texts = ["相同的测试文本"]
    
    cache_keys = {}
    for pool_type in ["avg", "cls", "last"]:
        # 创建临时参数
        temp_args = get_args([
            "--use_openai", "--api_key", "test", "--base_url", "http://test",
            "--model_name", "test", "--pool_type", pool_type,
            "--output_dir", "/tmp", "--task_list", "LEMBNeedleRetrieval",
            "--window_length_list", "256"
        ])
        temp_model = OpenAIEmbeddingModel(temp_args)
        temp_cached_model = task.create_cached_model_wrapper(temp_model)
        
        cache_key = task._get_comprehensive_cache_key(
            temp_cached_model, test_texts, "queries", 32
        )
        cache_keys[pool_type] = cache_key
        print(f"  {pool_type}: {cache_key[:16]}...")
    
    # 验证所有缓存键都不同
    unique_keys = set(cache_keys.values())
    if len(unique_keys) == len(cache_keys):
        print("  ✅ 不同pool_type生成了不同的缓存键")
    else:
        print("  ❌ 不同pool_type生成了相同的缓存键！")
        return False
    
    print("\n🎉 所有测试通过！pool_type传递链路完整且正确！")
    return True

if __name__ == "__main__":
    success = test_pool_type_integration()
    sys.exit(0 if success else 1) 