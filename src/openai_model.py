import os
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from mteb.evaluation.evaluators import DRESModel
from openai import OpenAI
# Add tiktoken import for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken is not installed, will use simple character estimation instead of token counting")

from utils import logger, get_chunked_docs


class OpenAIEmbeddingModel(DRESModel):
    """OpenAI SDK-based embedding model class, implementing MTEB's DRESModel interface"""
    
    def __init__(self, args):
        self.args = args
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.l2_norm = not args.no_l2_norm
        self.prefix_type = args.prefix_type
        self.prompt = args.prompt
        # 添加pool_type支持
        self.pool_type = getattr(args, 'pool_type', 'avg')
        self.max_retries = getattr(args, 'max_retries', 3)
        self.retry_delay = getattr(args, 'retry_delay', 1)
        # Add max_input_tokens parameter, reserve some space for special tokens
        self.max_input_tokens = getattr(args, 'max_input_tokens', 8192) - 10  # Reserve 10 tokens for special tokens
        
        # Initialize token encoder (if available)
        if TIKTOKEN_AVAILABLE:
            try:
                # Try to use a more suitable encoder
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Use base encoder
                logger.info("Using tiktoken cl100k_base encoder for token counting")
            except Exception as e:
                logger.warning(f"tiktoken initialization failed: {e}, will use character estimation")
                self.tokenizer = None
        else:
            self.tokenizer = None
            logger.warning("tiktoken not available, will use character estimation")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"Initialize OpenAI embedding model: {self.model_name}")
        logger.info(f"API Base URL: {self.base_url}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"L2 Norm: {self.l2_norm}")
        logger.info(f"Prefix Type: {self.prefix_type}")
        logger.info(f"Pool Type: {self.pool_type}")
        logger.info(f"Max Input Tokens (effective): {self.max_input_tokens}")

    def _truncate_text(self, text: str, prefix: str = "") -> str:
        """Truncate text to ensure it doesn't exceed token limit after adding prefix"""
        # Calculate tokens used by prefix
        prefix_tokens = 0
        if prefix:
            if self.tokenizer:
                try:
                    prefix_tokens = len(self.tokenizer.encode(prefix))
                except Exception as e:
                    logger.warning(f"prefix token counting failed: {e}, using character estimation")
                    prefix_tokens = len(prefix) // 3 + 2  # More conservative estimation
            else:
                # Use character estimation
                prefix_tokens = len(prefix) // 3 + 2  # More conservative estimation
        
        # Remaining available tokens
        available_tokens = max(10, self.max_input_tokens - prefix_tokens)  # Reserve at least 10 tokens
        
        logger.info(f"Truncation check: max_tokens={self.max_input_tokens}, prefix='{prefix}', prefix_tokens={prefix_tokens}, available_tokens={available_tokens}")
        
        if self.tokenizer:
            try:
                # Use tiktoken for precise token counting
                tokens = self.tokenizer.encode(text)
                logger.info(f"Original text token count: {len(tokens)}, needs truncation: {len(tokens) > available_tokens}")
                if len(tokens) > available_tokens:
                    # Truncate tokens and decode back to text
                    truncated_tokens = tokens[:available_tokens]
                    truncated_text = self.tokenizer.decode(truncated_tokens)
                    logger.info(f"Text truncated: original length {len(tokens)} tokens, truncated to {len(truncated_tokens)} tokens (reserved {prefix_tokens} tokens for prefix)")
                    return truncated_text
                return text
            except Exception as e:
                logger.warning(f"tiktoken encoding failed: {e}, using character estimation")
                # Fall back to character estimation
        
        # Use character estimation (more conservative estimate: 1 token ≈ 3 characters)
        estimated_tokens = len(text) // 3 + 1
        logger.info(f"Estimated text token count: {estimated_tokens}, needs truncation: {estimated_tokens > available_tokens}")
        if estimated_tokens > available_tokens:
            # Truncate text
            max_chars = available_tokens * 3
            truncated_text = text[:max_chars]
            logger.info(f"Text truncated: original length ~{estimated_tokens} tokens, truncated to ~{len(truncated_text)//3+1} tokens (reserved ~{prefix_tokens} tokens for prefix)")
            return truncated_text
        return text

    def encode_queries(self, queries: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        """Encode queries"""
        # In retrieval settings, queries are usually short, so no chunking needed
        batch_size = max(batch_size, 64)
        
        logger.info(f"Start encoding queries, count: {len(queries)}, prefix_type: {self.prefix_type}")
        
        # Determine prefix based on prefix_type, and truncate text before adding prefix
        if self.prefix_type == 'query_or_passage':
            prefix = 'query: '
            truncated_queries = [self._truncate_text(q, prefix) for q in queries]
            input_texts = [f'{prefix}{q}' for q in truncated_queries]
        else:
            prefix = self.prompt
            truncated_queries = [self._truncate_text(q, prefix) for q in queries]
            input_texts = [prefix + q for q in truncated_queries]
        
        # Final check: record token count of final input texts
        if self.tokenizer:
            for i, text in enumerate(input_texts[:3]):  # Only check first 3
                final_tokens = len(self.tokenizer.encode(text))
                logger.info(f"Final query text {i} token count: {final_tokens}")
        
        encoded_embeds: np.ndarray = self._do_encode(input_texts, batch_size)
        
        # For queries, according to original model logic, always normalize at the end
        # Since queries usually don't use chunking mode, _do_encode has already handled normalization
        # No additional processing needed here
        
        return encoded_embeds
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        """Encode corpus"""
        chunking_mode: str = os.getenv('CHUNKING_MODE')
        chunked_corpus: List[Dict[str, str]] = []
        chunked_index_list: List[tuple] = []
        
        logger.info(f"Start encoding corpus, count: {len(corpus)}, chunking_mode: {chunking_mode}, prefix_type: {self.prefix_type}")
        
        # Handle chunking
        if chunking_mode == 'chunk':
            doc_list = [doc['text'] for doc in corpus]
            chunked_doc_list, chunked_index_list = get_chunked_docs(self.args, doc_list)
            chunked_corpus = [{'text': doc} for doc in chunked_doc_list]
        else:
            chunked_corpus = corpus
        
        # Build input texts (merge title and text first)
        merged_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in chunked_corpus]
        
        # Determine prefix based on prefix_type, and truncate text before adding prefix
        if self.prefix_type == 'query_or_passage':
            prefix = 'passage: '
            truncated_texts = [self._truncate_text(t, prefix) for t in merged_texts]
            input_texts = [f'{prefix}{t}' for t in truncated_texts]
        elif self.prefix_type == 'nomic':
            prefix = 'search_document: '
            truncated_texts = [self._truncate_text(t, prefix) for t in merged_texts]
            input_texts = [f'{prefix}{t}' for t in truncated_texts]
        else:
            # For bge, none, instruction models, don't add prefix
            input_texts = [self._truncate_text(t) for t in merged_texts]
        
        # Final check: record token count of final input texts
        if self.tokenizer:
            for i, text in enumerate(input_texts[:3]):  # Only check first 3
                final_tokens = len(self.tokenizer.encode(text))
                logger.info(f"Final corpus text {i} token count: {final_tokens}")
        
        encoded_embeds: np.ndarray = self._do_encode(input_texts, batch_size)
        
        # If chunking was used, need to restore original document structure
        if chunking_mode == 'chunk':
            restored_embeds = []
            for st, ed in chunked_index_list:
                avg_embed = encoded_embeds[st:ed].mean(axis=0)
                if self.l2_norm:
                    avg_embed = avg_embed / np.linalg.norm(avg_embed)
                restored_embeds.append(avg_embed)
            encoded_embeds = np.array(restored_embeds)
        
        assert len(encoded_embeds) == len(corpus)
        return encoded_embeds
    
    def _do_encode(self, input_texts: List[str], batch_size: int) -> np.ndarray:
        """Perform actual encoding using OpenAI API"""
        encoded_embeds = []
        
        for start_idx in tqdm(range(0, len(input_texts), batch_size), desc='Encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]
            
            # Call OpenAI API to get embeddings
            batch_embeds = self._get_embeddings_with_retry(batch_input_texts)
            encoded_embeds.extend(batch_embeds)
        
        encoded_embeds = np.array(encoded_embeds)
        
        # Check chunking mode, handle normalization according to original model logic
        chunking_mode: str = os.getenv('CHUNKING_MODE')
        if self.l2_norm and chunking_mode != 'chunk':
            # Non-chunking mode: normalize directly
            encoded_embeds = encoded_embeds / np.linalg.norm(encoded_embeds, axis=1, keepdims=True)
        
        # Check for NaN values
        if np.isnan(encoded_embeds).any():
            logger.error('NaN values detected in encoded_embeds')
            exit(1)
        
        return encoded_embeds
    
    def _get_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """OpenAI API call with retry mechanism"""
        # Final safety check: validate and truncate again before API call
        final_texts = []
        for i, text in enumerate(texts):
            if self.tokenizer:
                try:
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) > self.max_input_tokens + 10:  # Add back reserved 10 tokens for checking
                        logger.warning(f"Text {i} still too long ({len(tokens)} tokens), performing final truncation")
                        # Emergency truncation to safe range
                        safe_tokens = tokens[:self.max_input_tokens]
                        text = self.tokenizer.decode(safe_tokens)
                        logger.info(f"Emergency truncation to {len(safe_tokens)} tokens")
                    final_texts.append(text)
                except Exception as e:
                    logger.warning(f"Final token check failed: {e}, using original text")
                    final_texts.append(text)
            else:
                # Use character estimation for final check
                estimated_tokens = len(text) // 3 + 1
                if estimated_tokens > self.max_input_tokens + 10:
                    logger.warning(f"Text {i} estimated still too long (~{estimated_tokens} tokens), performing final truncation")
                    safe_chars = (self.max_input_tokens - 5) * 3  # More conservative
                    text = text[:safe_chars]
                    logger.info(f"Emergency truncation to ~{len(text)//3+1} tokens")
                final_texts.append(text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=final_texts,
                    model=self.model_name
                )
                return [embedding.embedding for embedding in response.data]
            
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}")
                
                # If still a token limit error, perform more aggressive truncation
                if "maximum context length" in error_msg or "tokens" in error_msg:
                    if self.tokenizer and attempt == 0:  # Only execute on first retry
                        logger.warning("Token limit detected, performing more aggressive truncation")
                        final_texts = []
                        for text in texts:
                            try:
                                tokens = self.tokenizer.encode(text)
                                # More aggressive truncation: reduce 20% of tokens
                                safe_token_count = int(self.max_input_tokens * 0.8)
                                if len(tokens) > safe_token_count:
                                    safe_tokens = tokens[:safe_token_count]
                                    text = self.tokenizer.decode(safe_tokens)
                                    logger.info(f"Aggressive truncation to {len(safe_tokens)} tokens")
                                final_texts.append(text)
                            except:
                                # If tiktoken fails, use character truncation
                                safe_chars = int(self.max_input_tokens * 0.8 * 3)
                                final_texts.append(text[:safe_chars])
                        continue  # Retry API call
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"API call failed, maximum retries reached: {error_msg}")
                    raise
    
    def set_prompt(self, prompt: str):
        """Set prompt"""
        self.prompt = prompt 