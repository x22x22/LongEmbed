#!/bin/bash

# 默认配置
SKIP_MODEL_MANAGEMENT=false

# 定义要测试的模型列表 - 支持 ollama 和 vllm 两种类型
# 格式: "ollama|model" (ollama) 或 "vllm|pooling_type|model_name|model_code" (vllm)
MODELS=(
    "vllm|MEAN|intfloat/multilingual-e5-large|intfloat/multilingual-e5-large"
    "vllm|LAST|intfloat/multilingual-e5-large|intfloat/multilingual-e5-large"
    "vllm|CLS|intfloat/multilingual-e5-large|intfloat/multilingual-e5-large"
    "vllm|LAST|BAAI/bge-multilingual-gemma2|BAAI/bge-multilingual-gemma2"
    "vllm|MEAN|BAAI/bge-multilingual-gemma2|BAAI/bge-multilingual-gemma2"
    "vllm|CLS|BAAI/bge-multilingual-gemma2|BAAI/bge-multilingual-gemma2"
    "vllm|CLS|ibm-granite/granite-embedding-278m-multilingual|ibm-granite/granite-embedding-278m-multilingual"
    "vllm|MEAN|ibm-granite/granite-embedding-278m-multilingual|ibm-granite/granite-embedding-278m-multilingual"
    "vllm|LAST|ibm-granite/granite-embedding-278m-multilingual|ibm-granite/granite-embedding-278m-multilingual"
)

# 原有的 ollama 模型列表 (注释掉作为参考)
# OLD_MODELS=(
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
# BASE_URL="http://127.0.0.1:11434/v1"  # ollama 默认地址
BASE_URL="http://127.0.0.1:31090/v1"     # vllm 默认地址
OUTPUT_DIR="results"
BATCH_SIZE=12
MAX_INPUT_TOKENS=3072000
WINDOW_LENGTH_LIST="256 512 1024 2048 4096 8192 16384 32768"
TASK_LIST="LEMBNeedleRetrieval LEMBPasskeyRetrieval"
PREFIX_TYPE="query_or_passage"

# vLLM 配置
VLLM_PROJECT_PATH="/home/kdump/llm/project/vllm"
VLLM_SERVICE_SCRIPT="$VLLM_PROJECT_PATH/examples/online_serving/openai_embedding_long_text_service.sh"
VLLM_TMUX_SESSION=""  # 当前运行的tmux会话名
VLLM_LOG_DIR="./logs/vllm"
CONDA_ENV_NAME="vllm"  # 默认conda环境名

# 创建日志目录
LOGS_DIR="./logs"
mkdir -p "$LOGS_DIR"
mkdir -p "$VLLM_LOG_DIR"

# 日志文件
LOG_FILE="$LOGS_DIR/auto_test_$(date +%Y%m%d_%H%M%S).log"

# 中断标志
INTERRUPTED=false

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 解析模型配置的函数
parse_model_config() {
    local model_config="$1"
    local IFS='|'
    read -ra PARTS <<< "$model_config"
    
    RUNTIME="${PARTS[0]}"
    
    if [ "$RUNTIME" = "ollama" ]; then
        MODEL_NAME="${PARTS[1]}"
        POOLING_TYPE=""
        MODEL_CODE=""
    elif [ "$RUNTIME" = "vllm" ]; then
        POOLING_TYPE="${PARTS[1]}"
        MODEL_NAME="${PARTS[2]}"
        MODEL_CODE="${PARTS[3]}"
    else
        log "错误: 未知的模型类型: $RUNTIME"
        return 1
    fi
    
    return 0
}

# 获取模型显示名称
get_model_display_name() {
    local model_config="$1"
    parse_model_config "$model_config"
    
    if [ "$RUNTIME" = "ollama" ]; then
        echo "$MODEL_NAME"
    else
        echo "${MODEL_CODE:-$MODEL_NAME}"
    fi
}

# 获取输出目录
get_output_dir() {
    local model_config="$1"
    parse_model_config "$model_config"
    
    local display_name=$(get_model_display_name "$model_config")
    local safe_name="${display_name//[:\/]/_}"
    
    if [ "$RUNTIME" = "ollama" ]; then
        echo "${OUTPUT_DIR}/ollama/${safe_name}"
    else
        # vLLM 结果按 pooling_type 分组
        echo "${OUTPUT_DIR}/vllm/${POOLING_TYPE}/${safe_name}"
    fi
}

# 拉取所有模型的函数 (仅适用于 ollama)
pull_all_models() {
    log "开始拉取所有 ollama 模型"
    
    local success_count=0
    local total_count=0
    
    for model_config in "${MODELS[@]}"; do
        parse_model_config "$model_config"
        
        if [ "$RUNTIME" = "ollama" ]; then
            total_count=$((total_count + 1))
            log "正在拉取模型: $MODEL_NAME ($((success_count + 1))/$total_count)"
            
            if ollama pull "$MODEL_NAME"; then
                log "成功拉取模型: $MODEL_NAME"
                success_count=$((success_count + 1))
            else
                log "错误: 拉取模型失败: $MODEL_NAME"
            fi
            
            # 模型之间暂停一下，避免过度负载
            sleep 2
        else
            log "跳过 vLLM 模型: $MODEL_NAME (不需要拉取)"
        fi
    done
    
    log "ollama 模型拉取完成！成功拉取: $success_count/$total_count 个模型"
    
    if [ $total_count -gt 0 ] && [ $success_count -ne $total_count ]; then
        log "警告: 有 $((total_count - success_count)) 个 ollama 模型拉取失败"
        return 1
    fi
    
    return 0
}

# ====== ollama 相关函数 ======

# 检查 ollama 模型是否启动的函数
check_ollama_model_running() {
    local model_name="$1"
    ollama ps | grep -q "$model_name"
    return $?
}

# 启动 ollama 模型的函数
start_ollama_model() {
    local model_name="$1"
    log "正在启动 ollama 模型: $model_name"
    
    # 后台启动模型
    nohup ollama run "$model_name" > /dev/null 2>&1 &
    local ollama_pid=$!
    
    # 等待模型启动，最多等待5分钟
    local max_wait=300
    local wait_time=0
    local check_interval=10
    
    while [ $wait_time -lt $max_wait ]; do
        if check_ollama_model_running "$model_name"; then
            log "ollama 模型 $model_name 启动成功"
            return 0
        fi
        
        log "等待 ollama 模型启动... (已等待 ${wait_time}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    log "错误: ollama 模型 $model_name 启动超时"
    return 1
}

# 停止 ollama 模型的函数
stop_ollama_model() {
    local model_name="$1"
    log "正在停止 ollama 模型: $model_name"
    ollama stop "$model_name"
    
    # 等待模型完全停止
    sleep 5
    
    if ! check_ollama_model_running "$model_name"; then
        log "ollama 模型 $model_name 已成功停止"
        return 0
    else
        log "警告: ollama 模型 $model_name 可能未完全停止"
        return 1
    fi
}

# ====== vLLM 相关函数 ======

# 检查 vLLM 服务是否在运行
check_vllm_service_running() {
    local port="31090"
    curl -s "http://localhost:$port/health" >/dev/null 2>&1
    return $?
}

# 启动 vLLM 服务的函数
start_vllm_service() {
    local pooling_type="$1"
    local model_name="$2"
    local model_code="$3"
    
    log "正在启动 vLLM 服务: $model_name (pooling: $pooling_type)"
    
    # 生成唯一的tmux会话名
    local session_name="vllm_$(date +%Y%m%d_%H%M%S)_$$"
    VLLM_TMUX_SESSION="$session_name"
    
    # 检查tmux是否可用
    if ! command -v tmux &> /dev/null; then
        log "错误: 未找到 tmux 命令，请先安装 tmux"
        return 1
    fi
    
    # 检查会话是否已存在
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log "警告: tmux 会话 $session_name 已存在，正在终止旧会话"
        tmux kill-session -t "$session_name"
        sleep 2
    fi
    
    # 生成唯一的日志文件名（使用绝对路径）
    local current_dir=$(pwd)
    local log_file="$current_dir/$VLLM_LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"
    
    # 确保日志目录存在
    mkdir -p "$(dirname "$log_file")"
    
    # 创建新的tmux会话并启动vLLM服务
    log "创建 tmux 会话: $session_name"
    log "使用conda环境: $CONDA_ENV_NAME"
    log "日志文件: $log_file"
    
    # 创建启动脚本
    local start_script="$current_dir/$VLLM_LOG_DIR/start_vllm_$(date +%Y%m%d_%H%M%S).sh"
    cat > "$start_script" << EOF
#!/bin/bash
export POOLING_TYPE='$pooling_type'
export MODEL_NAME='$model_name'
export MODEL_CODE='$model_code'
export API_KEY='$API_KEY'

echo "=========================================="
echo "启动 vLLM 服务"
echo "模型: \$MODEL_NAME"
echo "代码: \$MODEL_CODE"
echo "池化类型: \$POOLING_TYPE"
echo "时间: \$(date)"
echo "=========================================="

# 直接激活 conda 环境并运行，确保输出正确传递
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate '$CONDA_ENV_NAME'

cd '$VLLM_PROJECT_PATH'

# 运行 vLLM 服务脚本，确保所有输出都被捕获
exec bash '$VLLM_SERVICE_SCRIPT' 2>&1 | tee '$log_file'
EOF
    chmod +x "$start_script"
    
    # 在tmux会话中启动vLLM服务
    tmux new-session -d -s "$session_name" "bash '$start_script'"
    
    if [ $? -ne 0 ]; then
        log "错误: 创建 tmux 会话失败"
        return 1
    fi
    
    log "vLLM 服务已在 tmux 会话中启动: $session_name"
    log "可使用 'tmux attach -t $session_name' 查看实时输出"
    
    # 等待服务启动，最多等待10分钟
    local max_wait=600
    local wait_time=0
    local check_interval=15
    
    while [ $wait_time -lt $max_wait ]; do
        # 检查tmux会话是否还存在
        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            log "错误: tmux 会话意外退出，请检查日志: $log_file"
            return 1
        fi
        
        if check_vllm_service_running; then
            log "vLLM 服务启动成功，模型: $model_name"
            return 0
        fi
        
        log "等待 vLLM 服务启动... (已等待 ${wait_time}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    log "错误: vLLM 服务启动超时"
    return 1
}

# 停止 vLLM 服务的函数
stop_vllm_service() {
    log "正在停止 vLLM 服务"
    
    # 如果有活动的tmux会话，则终止它
    if [ -n "$VLLM_TMUX_SESSION" ]; then
        if tmux has-session -t "$VLLM_TMUX_SESSION" 2>/dev/null; then
            log "正在终止 tmux 会话: $VLLM_TMUX_SESSION"
            tmux kill-session -t "$VLLM_TMUX_SESSION"
            
            # 等待会话完全终止
            local wait_time=0
            local max_wait=30
            while [ $wait_time -lt $max_wait ] && tmux has-session -t "$VLLM_TMUX_SESSION" 2>/dev/null; do
                sleep 1
                wait_time=$((wait_time + 1))
            done
            
            if tmux has-session -t "$VLLM_TMUX_SESSION" 2>/dev/null; then
                log "警告: tmux 会话可能未完全终止"
            else
                log "tmux 会话已成功终止: $VLLM_TMUX_SESSION"
            fi
        else
            log "tmux 会话不存在或已终止: $VLLM_TMUX_SESSION"
        fi
        
        # 清空会话名
        VLLM_TMUX_SESSION=""
    fi
    
    # 清理所有与vllm相关的tmux会话（作为安全措施）
    local vllm_sessions=$(tmux list-sessions 2>/dev/null | grep "^vllm_" | cut -d: -f1 || true)
    if [ -n "$vllm_sessions" ]; then
        log "发现残留的 vLLM tmux 会话，正在清理: $vllm_sessions"
        echo "$vllm_sessions" | while read -r session; do
            if [ -n "$session" ]; then
                tmux kill-session -t "$session" 2>/dev/null || true
                log "已清理 tmux 会话: $session"
            fi
        done
    fi
    
    # 清理临时启动脚本
    if [ -d "$VLLM_LOG_DIR" ]; then
        local old_scripts=$(find "$VLLM_LOG_DIR" -name "start_vllm_*.sh" -mtime +1 2>/dev/null || true)
        if [ -n "$old_scripts" ]; then
            log "清理旧的启动脚本"
            echo "$old_scripts" | xargs -r rm -f
        fi
    fi
    
    # 额外检查是否还有 vLLM 进程在运行（兜底措施）
    local remaining_pids=$(pgrep -f "vllm serve" || true)
    if [ -n "$remaining_pids" ]; then
        log "发现残留的 vLLM 进程，正在清理: $remaining_pids"
        echo "$remaining_pids" | xargs -r kill -TERM
        sleep 5
        echo "$remaining_pids" | xargs -r kill -KILL 2>/dev/null || true
    fi
    
    log "vLLM 服务已停止"
    return 0
}

# ====== 通用模型管理函数 ======

# 启动模型的函数
start_model() {
    local model_config="$1"
    parse_model_config "$model_config"
    
    if [ "$RUNTIME" = "ollama" ]; then
        start_ollama_model "$MODEL_NAME"
    elif [ "$RUNTIME" = "vllm" ]; then
        start_vllm_service "$POOLING_TYPE" "$MODEL_NAME" "$MODEL_CODE"
    else
        log "错误: 未知的模型类型: $RUNTIME"
        return 1
    fi
}

# 停止模型的函数
stop_model() {
    local model_config="$1"
    parse_model_config "$model_config"
    
    if [ "$RUNTIME" = "ollama" ]; then
        stop_ollama_model "$MODEL_NAME"
    elif [ "$RUNTIME" = "vllm" ]; then
        stop_vllm_service
    else
        log "错误: 未知的模型类型: $RUNTIME"
        return 1
    fi
}

# 运行测试的函数
run_test() {
    local model_config="$1"
    parse_model_config "$model_config"
    
    local test_output_dir=$(get_output_dir "$model_config")
    local display_name=$(get_model_display_name "$model_config")
    
    # 根据模型类型设置 BASE_URL
    local test_base_url
    if [ "$RUNTIME" = "ollama" ]; then
        test_base_url="http://127.0.0.1:11434/v1"
    else
        test_base_url="http://127.0.0.1:31090/v1"
    fi
    
    log "开始测试模型: $display_name (类型: $RUNTIME)"
    log "测试结果将保存到: $test_output_dir"
    log "使用 API 地址: $test_base_url"
    
    # 确保输出目录存在
    mkdir -p "$test_output_dir"
    
    # 根据模型类型使用不同的模型名称
    local test_model_name
    if [ "$RUNTIME" = "ollama" ]; then
        test_model_name="$MODEL_NAME"
    else
        test_model_name="$MODEL_CODE"
    fi
    
    # 准备pool_type参数
    local pool_type_arg=""
    if [ "$RUNTIME" = "vllm" ] && [ -n "$POOLING_TYPE" ]; then
        # 将vLLM的POOLING_TYPE转换为对应的pool_type值
        case "$POOLING_TYPE" in
            "MEAN")
                pool_type_arg="--pool_type avg"
                ;;
            "CLS")
                pool_type_arg="--pool_type cls"
                ;;
            "LAST")
                pool_type_arg="--pool_type last"
                ;;
            *)
                log "警告: 未知的POOLING_TYPE: $POOLING_TYPE，使用默认pool_type"
                pool_type_arg=""
                ;;
        esac
        log "使用pool_type参数: $pool_type_arg"
    fi
    
    python src/test_openai_long_embed_local.py \
        --use_openai \
        --api_key "$API_KEY" \
        --base_url "$test_base_url" \
        --model_name "$test_model_name" \
        --output_dir "$test_output_dir" \
        --batch_size $BATCH_SIZE \
        --max_input_tokens $MAX_INPUT_TOKENS \
        --window_length_list $WINDOW_LENGTH_LIST \
        --task_list $TASK_LIST \
        --prefix_type "$PREFIX_TYPE" \
        $pool_type_arg
    
    local test_result=$?
    
    if [ $test_result -eq 0 ]; then
        log "模型 $display_name 测试完成"
        return 0
    else
        log "错误: 模型 $display_name 测试失败"
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
    mkdir -p "$OUTPUT_DIR/ollama"
    
    # 为每个 vLLM 模型的 pooling_type 创建目录
    for model_config in "${MODELS[@]}"; do
        parse_model_config "$model_config"
        if [ "$RUNTIME" = "vllm" ]; then
            mkdir -p "$OUTPUT_DIR/vllm/$POOLING_TYPE"
        fi
    done
    
    # 循环测试每个模型
    for model_config in "${MODELS[@]}"; do
        # 检查是否被中断
        if [ "$INTERRUPTED" = true ]; then
            log "检测到中断信号，停止处理剩余模型"
            break
        fi
        
        log "========================================"
        
        local display_name=$(get_model_display_name "$model_config")
        log "开始处理模型: $display_name ($((success_count + 1))/$total_count)"
        
        parse_model_config "$model_config"
        log "模型类型: $RUNTIME"
        
        # 步骤1: 启动模型
        if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
            # 检查中断标志
            if [ "$INTERRUPTED" = true ]; then
                log "检测到中断信号，跳过模型启动"
                break
            fi
            
            if start_model "$model_config"; then
                # 再次检查中断标志
                if [ "$INTERRUPTED" = true ]; then
                    log "检测到中断信号，停止当前模型并退出"
                    stop_model "$model_config"
                    break
                fi
                
                # 步骤2: 运行测试
                if run_test "$model_config"; then
                    success_count=$((success_count + 1))
                    log "模型 $display_name 处理成功"
                else
                    log "模型 $display_name 测试失败"
                fi
            else
                log "模型 $display_name 启动失败，跳过测试"
            fi
        else
            log "跳过模型管理，直接运行测试: $display_name"
            
            # 检查中断标志
            if [ "$INTERRUPTED" = true ]; then
                log "检测到中断信号，跳过测试"
                break
            fi
            
            if run_test "$model_config"; then
                success_count=$((success_count + 1))
                log "模型 $display_name 处理成功"
            else
                log "模型 $display_name 测试失败"
            fi
        fi
        
        # 检查中断标志（在停止模型前）
        if [ "$INTERRUPTED" = true ]; then
            log "检测到中断信号，即将退出"
            if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
                stop_model "$model_config"
            fi
            break
        fi
        
        # 步骤3: 停止模型
        if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
            stop_model "$model_config"
        fi
        
        log "模型 $display_name 处理完成"
        log "========================================"
        
        # 模型之间暂停一下
        sleep 10
    done
    
    # 根据是否被中断显示不同的完成信息
    if [ "$INTERRUPTED" = true ]; then
        log "测试被用户中断！"
        log "已完成测试: $success_count 个模型（共计划 $total_count 个）"
        log "详细日志请查看: $LOG_FILE"
    else
        log "所有测试完成！"
        log "成功测试: $success_count/$total_count 个模型"
        log "详细日志请查看: $LOG_FILE"
    fi
    
    # 生成测试报告
    generate_report
    
    # 如果被中断，使用特殊退出码
    if [ "$INTERRUPTED" = true ]; then
        log "脚本因用户中断而退出"
        exit 130  # 130 是 Ctrl+C 的标准退出码
    fi
}

# 生成测试报告
generate_report() {
    local report_file="$LOGS_DIR/test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "=== 自动化测试报告 ===" > "$report_file"
    echo "测试时间: $(date)" >> "$report_file"
    echo "计划测试模型数量: ${#MODELS[@]}" >> "$report_file"
    if [ "$INTERRUPTED" = true ]; then
        echo "测试状态: 被用户中断" >> "$report_file"
        echo "实际完成测试: $success_count 个模型" >> "$report_file"
    else
        echo "测试状态: 正常完成" >> "$report_file"
        echo "成功完成测试: $success_count 个模型" >> "$report_file"
    fi
    if [ "$SKIP_MODEL_MANAGEMENT" = true ]; then
        echo "模型管理模式: 跳过（假设模型已运行）" >> "$report_file"
    else
        echo "模型管理模式: 自动启动/停止" >> "$report_file"
    fi
    echo "" >> "$report_file"
    
    echo "测试的模型列表:" >> "$report_file"
    for i in "${!MODELS[@]}"; do
        local display_name=$(get_model_display_name "${MODELS[i]}")
        parse_model_config "${MODELS[i]}"
        echo "$((i + 1)). $display_name (类型: $RUNTIME)" >> "$report_file"
    done
    
    echo "" >> "$report_file"
    echo "结果目录结构:" >> "$report_file"
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -type d | head -20 >> "$report_file"
    fi
    
    log "测试报告已生成: $report_file"
}

# 用户中断处理函数
handle_interrupt() {
    log ""
    log "接收到中断信号 (Ctrl+C)，正在进行优雅停机..."
    log "当前操作完成后将停止处理剩余模型"
    INTERRUPTED=true
}

# 脚本退出时的清理函数
cleanup() {
    log "脚本退出，正在清理..."
    
    if [ "$SKIP_MODEL_MANAGEMENT" = false ]; then
        # 停止所有可能还在运行的模型
        for model_config in "${MODELS[@]}"; do
            parse_model_config "$model_config"
            
            if [ "$RUNTIME" = "ollama" ]; then
                if check_ollama_model_running "$MODEL_NAME"; then
                    log "清理: 停止 ollama 模型 $MODEL_NAME"
                    ollama stop "$MODEL_NAME" 2>/dev/null
                fi
            elif [ "$RUNTIME" = "vllm" ]; then
                log "清理: 停止 vLLM 服务"
                stop_vllm_service
            fi
        done
    else
        log "跳过模型管理模式，不停止模型"
    fi
}

# 设置信号处理
trap cleanup EXIT TERM
trap handle_interrupt INT

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-model-management|--no-model-control)
            SKIP_MODEL_MANAGEMENT=true
            shift
            ;;
        --pull-models)
            echo "开始拉取所有 ollama 模型..."
            pull_all_models
            exit $?
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --skip-model-management, --no-model-control    跳过模型的启动和停止，假设模型已经运行"
            echo "  --pull-models                                  拉取所有 ollama 模型然后退出"
            echo "  --help, -h                                     显示此帮助信息"
            echo ""
            echo "模型配置格式:"
            echo "  ollama: \"ollama|model_name\""
            echo "  vllm:   \"vllm|pooling_type|model_name|model_code\""
            echo ""
            echo "环境配置:"
            echo "  vLLM 将在conda环境 '$CONDA_ENV_NAME' 中运行"
            echo "  vLLM 服务使用 tmux 会话管理（需要安装 tmux）"
            echo "  如需修改，请在脚本中更改 CONDA_ENV_NAME 变量"
            echo ""
            echo "输出目录:"
            echo "  ollama 模型: ./results/ollama/{model_name}/"
            echo "  vllm 模型:   ./results/vllm/{pooling_type}/{model_name}/"
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
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 python 命令"
    exit 1
fi

if [ ! -f "src/test_openai_long_embed_local.py" ]; then
    echo "错误: 未找到测试脚本 src/test_openai_long_embed_local.py"
    exit 1
fi

# 检查是否有 ollama 模型，如果有则需要 ollama 命令
has_ollama_models=false
for model_config in "${MODELS[@]}"; do
    parse_model_config "$model_config"
    if [ "$RUNTIME" = "ollama" ]; then
        has_ollama_models=true
        break
    fi
done

if [ "$has_ollama_models" = true ] && ! command -v ollama &> /dev/null; then
    echo "错误: 配置了 ollama 模型但未找到 ollama 命令，请确保 ollama 已安装"
    exit 1
fi

# 检查是否有 vllm 模型，如果有则检查 vllm 服务脚本
has_vllm_models=false
for model_config in "${MODELS[@]}"; do
    parse_model_config "$model_config"
    if [ "$RUNTIME" = "vllm" ]; then
        has_vllm_models=true
        break
    fi
done

if [ "$has_vllm_models" = true ] && [ ! -f "$VLLM_SERVICE_SCRIPT" ]; then
    echo "错误: 配置了 vllm 模型但未找到 vllm 服务脚本: $VLLM_SERVICE_SCRIPT"
    exit 1
fi

# 检查tmux依赖（仅在有vllm模型时需要）
if [ "$has_vllm_models" = true ] && ! command -v tmux &> /dev/null; then
    echo "错误: 配置了 vllm 模型但未找到 tmux 命令，请先安装 tmux"
    echo "在Ubuntu/Debian系统中可使用: sudo apt-get install tmux"
    echo "在CentOS/RHEL系统中可使用: sudo yum install tmux"
    exit 1
fi

# 执行主函数
main 