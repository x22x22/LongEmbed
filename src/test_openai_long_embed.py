import os
import json
import logging

from mteb import MTEB

from utils import logger, get_args
from encoder_model import RetrievalModel
from openai_model import OpenAIEmbeddingModel

logging.getLogger().setLevel(logging.INFO)

def main():
    args = get_args()
    
    # Choose model based on whether using OpenAI
    if args.use_openai:
        if not args.api_key or not args.base_url or not args.model_name:
            logger.error("When using OpenAI mode, api_key, base_url and model_name parameters must be provided")
            exit(1)
        
        logger.info("Using OpenAI embedding model")
        model = OpenAIEmbeddingModel(args)
        model_name = args.model_name
    else:
        if not args.model_name_or_path:
            logger.error("When using local mode, model_name_or_path parameter must be provided")
            exit(1)
        
        logger.info("Using local embedding model")
        model = RetrievalModel(args)
        model_name = os.path.basename(os.path.normpath(args.model_name_or_path))

    # Set output directory
    mteb_output_dir = os.path.join(args.output_dir, model_name)

    # Handle chunking mode
    chunking_mode: str = os.getenv('CHUNKING_MODE')
    if chunking_mode != "no_chunk":
        chunk_max_len = os.getenv('MAX_TOKEN_NUM', "0")
        mteb_output_dir += f"_{chunking_mode}-{chunk_max_len}"
    
    # Handle position mode (only applies to local models)
    if not args.use_openai:
        if args.pos_mode != "original":
            mteb_output_dir += f'_{args.pos_mode}'
        if args.use_self_extend == True:
            mteb_output_dir += f"_se_{model.encode_max_length}"
        if args.rope_theta != 10000:
            mteb_output_dir += f"_theta{args.rope_theta}_{model.encode_max_length}"
        if args.rotary_scaling_factor != None:
            mteb_output_dir += f"_rsf{args.rotary_scaling_factor}"
    
    # If using OpenAI, add identifier
    if args.use_openai:
        mteb_output_dir += "_openai"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(mteb_output_dir, exist_ok=True)

    retrieval_task_list = []
    needle_passkey_task_list = []
    output_dict = dict()
    needle_passkey_score_list = list()

    # Filter tasks
    for task in ["LEMBSummScreenFDRetrieval", "LEMBQMSumRetrieval","LEMBWikimQARetrieval","LEMBNarrativeQARetrieval"]:
        if task in args.task_list:
            retrieval_task_list.append(task)
    
    for task in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        if task in args.task_list:
            needle_passkey_task_list.append(task)

    # Evaluate needle and passkey retrieval tasks
    if needle_passkey_task_list != []:
        logger.info(f"Start evaluating needle and passkey tasks: {needle_passkey_task_list}")
        
        context_length_list = list(args.window_length_list)
        context_length_list.sort()

        evaluation = MTEB(tasks=needle_passkey_task_list)
        results = evaluation.run(model, output_folder=mteb_output_dir, overwrite_results=False, batch_size=args.batch_size, verbosity=0)
        
        for key, value in results.items():
            needle_passkey_score_list = []
            for ctx_len in context_length_list:
                needle_passkey_score_list.append([ctx_len, value[f"test_{ctx_len}"]["ndcg_at_1"]])
            needle_passkey_score_list.append(["avg", sum([x[1] for x in needle_passkey_score_list])/len(context_length_list)])
            output_dict[key] = {item[0]: item[1] for item in needle_passkey_score_list}

    # Evaluate retrieval tasks
    if retrieval_task_list != []:
        logger.info(f"Start evaluating retrieval tasks: {retrieval_task_list}")
        
        evaluation = MTEB(tasks=retrieval_task_list)
        results = evaluation.run(model, output_folder=mteb_output_dir, overwrite_results=False, batch_size=args.batch_size, verbosity=0)

        for key, value in results.items():
            split = "test" if "test" in value else "validation"
            output_dict[key] = {"ndcg@1": value[split]["ndcg_at_1"], "ndcg@10": value[split]["ndcg_at_10"]}
        
    logger.info("Evaluation results:")
    logger.info(json.dumps(output_dict, indent=2, ensure_ascii=False))

    # Save results
    # if len(args.task_list) == 6:
    results_file = os.path.join(mteb_output_dir, 'overall_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main() 