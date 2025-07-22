#!/bin/bash

# OpenAI embedding model LongEmbed evaluation script

# Set basic parameters
API_KEY="your-api-key"  # Replace with your actual API key
BASE_URL="http://10.22.0.188:31090/v1"
# MODEL_NAME="jina-embeddings-v3"
# MODEL_NAME="bge-m3"
MODEL_NAME="multilingual-e5-large"
# MODEL_NAME="e5-base-4k"
# MODEL_NAME="nomic-embed-text-v1.5"
# MODEL_NAME="Qwen3-Embedding-0.6B"
# MODEL_NAME="jina-embeddings-v4-vllm-retrieval"
# MODEL_NAME="jina-embeddings-v4"
# MODEL_NAME="gte-Qwen2-1.5B-instruct"
# MODEL_NAME="Qwen3-Embedding-4B"

# Set output directory
OUTPUT_DIR="results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Set Python path
export PYTHONPATH=$PYTHONPATH:src

# Evaluate all tasks
echo "Starting evaluation of OpenAI embedding model: $MODEL_NAME"

# Run evaluation
# Note: jina-embeddings-v3 model has maximum context length of 8194 tokens, using conservative 8000 to ensure no limit exceeded
    # --window_length_list 2048 4096 8192 16384 32768 \
    # --window_length_list 256 512 1024 2048 4096 8192 16384 32768 65536 \
    # --task_list "LEMBSummScreenFDRetrieval" "LEMBQMSumRetrieval" "LEMBWikimQARetrieval" "LEMBNarrativeQARetrieval" "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval" \
# python src/test_openai_long_embed.py \
#     --use_openai \
#     --api_key "$API_KEY" \
#     --base_url "$BASE_URL" \
#     --model_name "$MODEL_NAME" \
#     --output_dir "$OUTPUT_DIR" \
#     --batch_size 40 \
#     --max_input_tokens 8000 \
#     --window_length_list 256 512 1024 2048 4096 8192 \
#     --task_list "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval" \
#     --prefix_type "query_or_passage"

# python src/test_openai_long_embed.py \
#     --use_openai \
#     --api_key "$API_KEY" \
#     --base_url "$BASE_URL" \
#     --model_name "$MODEL_NAME" \
#     --output_dir "$OUTPUT_DIR" \
#     --batch_size 40 \
#     --max_input_tokens 450 \
#     --window_length_list 256 512 \
#     --task_list "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval" \
#     --prefix_type "query_or_passage"

# python src/test_openai_long_embed.py \
#     --use_openai \
#     --api_key "$API_KEY" \
#     --base_url "$BASE_URL" \
#     --model_name "$MODEL_NAME" \
#     --output_dir "$OUTPUT_DIR" \
#     --batch_size 40 \
#     --max_input_tokens 3072000 \
#     --window_length_list 256 512 1024 2048 4096 8192 16384 32768 65536 \
#     --task_list "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval" \
#     --prefix_type "query_or_passage"

    # --no_l2_norm \

python src/test_openai_long_embed.py \
    --use_openai \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 40 \
    --max_input_tokens 3072000 \
    --window_length_list 512 1024 2048 4096 8192 \
    --task_list "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval" \
    --prefix_type "query_or_passage"

echo "Evaluation completed! Results saved in $OUTPUT_DIR directory"
echo "Note: If chunking strategy was used to handle long texts, there will be corresponding hints in the logs"
