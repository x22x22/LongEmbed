#!/bin/bash

# 默认配置
SKIP_MODEL_MANAGEMENT=true

# 定义要测试的模型列表
MODELS=(
    "vllm/multilingual-e5-large"
)

# MODELS=(
#     "zylonai/multilingual-e5-large:latest"
#     "qllama/multilingual-e5-large:q4_k_m"
#     "qllama/multilingual-e5-large:f16"
#     "jeffh/intfloat-multilingual-e5-large:q8_0"
#     "jeffh/intfloat-multilingual-e5-large:f16"
#     "jeffh/intfloat-multilingual-e5-large:f32"
#     "dengcao/Qwen3-Embedding-0.6B:Q8_0"
#     "dengcao/Qwen3-Embedding-0.6B:F16"
#     "dengcao/Qwen3-Embedding-4B:Q4_K_M"
#     "dengcao/Qwen3-Embedding-4B:F16"
#     "dengcao/Qwen3-Embedding-8B:Q4_K_M"
#     "dengcao/Qwen3-Embedding-8B:F16"
#     "qllama/bge-m3:q4_k_m"
#     "bge-m3:567m-fp16"
# )

# 配置参数
API_KEY="your-api-key"
# BASE_URL="http://127.0.0.1:11434/v1"
BASE_URL="http://127.0.0.1:31090/v1"
OUTPUT_DIR="results"
BATCH_SIZE=12
MAX_INPUT_TOKENS=3072000
WINDOW_LENGTH_LIST="256 512 1024 2048 4096 8192 16384 32768"
TASK_LIST="LEMBNeedleRetrieval LEMBPasskeyRetrieval"
PREFIX_TYPE="query_or_passage"

# 创建日志目录
LOGS_DIR="./logs"
mkdir -p "$LOGS_DIR"

# 日志文件
LOG_FILE="$LOGS_DIR/auto_test_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 拉取所有模型的函数
pull_all_models() {
    log "开始拉取所有模型，共 ${#MODELS[@]} 个模型"
    
    local success_count=0
    local total_count=${#MODELS[@]}
    
    for model in "${MODELS[@]}"; do
        log "正在拉取模型: $model ($((success_count + 1))/$total_count)"
        
        if ollama pull "$model"; then
            log "成功拉取模型: $model"
            success_count=$((success_count + 1))
        else
            log "错误: 拉取模型失败: $model"
        fi
        
        # 模型之间暂停一下，避免过度负载
        sleep 2
    done
    
    log "模型拉取完成！成功拉取: $success_count/$total_count 个模型"
    
    if [ $success_count -ne $total_count ]; then
        log "警告: 有 $((total_count - success_count)) 个模型拉取失败"
        return 1
    fi
    
    return 0
}

# 检查模型是否启动的函数
check_model_running() {
    local model_name="$1"
    ollama ps | grep -q "$model_name"
    return $?
}

# 启动模型的函数
start_model() {
    local model_name="$1"
    log "正在启动模型: $model_name"
    
    # 后台启动模型
    nohup ollama run "$model_name" > /dev/null 2>&1 &
    local ollama_pid=$!
    
    # 等待模型启动，最多等待5分钟
    local max_wait=300
    local wait_time=0
    local check_interval=10
    
    while [ $wait_time -lt $max_wait ]; do
        if check_model_running "$model_name"; then
            log "模型 $model_name 启动成功"
            return 0
        fi
        
        log "等待模型启动... (已等待 ${wait_time}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    log "错误: 模型 $model_name 启动超时"
    return 1
}

# 停止模型的函数
stop_model() {
    local model_name="$1"
    log "正在停止模型: $model_name"
    ollama stop "$model_name"
    
    # 等待模型完全停止
    sleep 5
    
    if ! check_model_running "$model_name"; then
        log "模型 $model_name 已成功停止"
        return 0
    else
        log "警告: 模型 $model_name 可能未完全停止"
        return 1
    fi
}

# 运行测试的函数
run_test() {
    local model_name="$1"
    # 使用完整的模型名称，不截取
    local test_output_dir="${OUTPUT_DIR}/${model_name//[:\/]/_}"
    
    log "开始测试模型: $model_name"
    log "测试结果将保存到: $test_output_dir"
    
    python src/test_openai_long_embed_local.py \
        --use_openai \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --model_name "$model_name" \
        --output_dir "$test_output_dir" \
        --batch_size $BATCH_SIZE \
        --max_input_tokens $MAX_INPUT_TOKENS \
        --window_length_list $WINDOW_LENGTH_LIST \
        --task_list $TASK_LIST \
        --prefix_type "$PREFIX_TYPE"
    
    local test_result=$?
    
    if [ $test_result -eq 0 ]; then
        log "模型 $model_name 测试完成"
        return 0
    else
        log "错误: 模型 $model_name 测试失败"
        return 1
    fi
}

# 主函数
main() {
    if [ "$SKIP_MODEL_MANAGEMENT" = true ]; then
        log "开始自动化测试，共 ${#MODELS[@]} 个模型（跳过模型管理）"
        log "注意：假设所需模型已经在运行中"
    else
        log "开始自动化测试，共 ${#MODELS[@]} 个模型（包含模型管理）"
    fi
    
    local success_count=0
    local total_count=${#MODELS[@]}
    
    # 确保结果目录存在
    mkdir -p "$OUTPUT_DIR"
    
    # 循环测试每个模型
    for model in "${MODELS[@]}"; do
        log "========================================"
        log "开始处理模型: $model ($((success_count + 1))/$total_count)"
        
        # 步骤1: 启动模型
        if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
            if start_model "$model"; then
                # 步骤2: 运行测试
                if run_test "$model"; then
                    success_count=$((success_count + 1))
                    log "模型 $model 处理成功"
                else
                    log "模型 $model 测试失败"
                fi
            else
                log "模型 $model 启动失败，跳过测试"
            fi
        else
            log "跳过模型管理，直接运行测试: $model"
            if run_test "$model"; then
                success_count=$((success_count + 1))
                log "模型 $model 处理成功"
            else
                log "模型 $model 测试失败"
            fi
        fi
        
        # 步骤3: 停止模型
        if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
            stop_model "$model"
        fi
        
        log "模型 $model 处理完成"
        log "========================================"
        
        # 模型之间暂停一下
        sleep 10
    done
    
    log "所有测试完成！"
    log "成功测试: $success_count/$total_count 个模型"
    log "详细日志请查看: $LOG_FILE"
    
    # 生成测试报告
    generate_report
}

# 生成测试报告
generate_report() {
    local report_file="$LOGS_DIR/test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "=== 自动化测试报告 ===" > "$report_file"
    echo "测试时间: $(date)" >> "$report_file"
    echo "测试模型数量: ${#MODELS[@]}" >> "$report_file"
    if [ "$SKIP_MODEL_MANAGEMENT" = true ]; then
        echo "模型管理模式: 跳过（假设模型已运行）" >> "$report_file"
    else
        echo "模型管理模式: 自动启动/停止" >> "$report_file"
    fi
    echo "" >> "$report_file"
    
    echo "测试的模型列表:" >> "$report_file"
    for i in "${!MODELS[@]}"; do
        echo "$((i + 1)). ${MODELS[i]}" >> "$report_file"
    done
    
    echo "" >> "$report_file"
    echo "结果目录结构:" >> "$report_file"
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -type d | head -20 >> "$report_file"
    fi
    
    log "测试报告已生成: $report_file"
}

# 脚本退出时的清理函数
cleanup() {
    log "脚本中断，正在清理..."
    
    if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
        # 停止所有可能还在运行的模型
        for model in "${MODELS[@]}"; do
            if check_model_running "$model"; then
                log "清理: 停止模型 $model"
                ollama stop "$model" 2>/dev/null
            fi
        done
    else
        log "跳过模型管理模式，不停止模型"
    fi
}

# 设置信号处理
trap cleanup EXIT INT TERM

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-model-management|--no-model-control)
            SKIP_MODEL_MANAGEMENT=true
            shift
            ;;
        --pull-models)
            echo "开始拉取所有模型..."
            pull_all_models
            exit $?
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --skip-model-management, --no-model-control    跳过模型的启动和停止，假设模型已经运行"
            echo "  --pull-models                                  拉取所有模型然后退出"
            echo "  --help, -h                                     显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# 检查依赖
if ! command -v ollama &> /dev/null; then
    echo "错误: 未找到 ollama 命令，请确保 ollama 已安装"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "错误: 未找到 python 命令"
    exit 1
fi

if [ ! -f "src/test_openai_long_embed_local.py" ]; then
    echo "错误: 未找到测试脚本 src/test_openai_long_embed_local.py"
    exit 1
fi

# 执行主函数
main 