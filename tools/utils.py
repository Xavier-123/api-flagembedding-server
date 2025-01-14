import os
import gc
import uuid
import torch
from typing import List
from pydantic import BaseModel, Field
from FlagEmbedding import FlagAutoReranker, FlagAutoModel
from transformers import AutoTokenizer

from tools.log import logger

curr_path = os.path.split(os.path.abspath(__file__))[0]


def get_devices():
    # 判断显卡环境
    if torch.cuda.is_available():
        devices = "cuda:0"
    else:
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                torch.npu.set_compile_mode(jit_compile=False)
                option = {"NPU_FUZZY_COMPILE_BLACKLIST": "Tril"}
                torch.npu.set_option(option)
                devices = "npu"
            else:
                devices = "cpu"
        except Exception as e:
            devices = "cpu"
    return devices


devices = get_devices()

# 初始化embedd
embed_model_dict = {}
embed_tokenizer_dict = {}
model_name_or_path = r'F:\inspur\EMBEDDING_MODEL\AI-ModelScope\bge-small-zh-v1.5'
# model_name_or_path = os.environ.get("EMBEDDING_PATH", "")
if model_name_or_path == "":
    logger.info("Please set the environment variable 'EMBEDDING_PATH'")
base_embed_model_name = os.path.basename(model_name_or_path)
embed_model = FlagAutoModel.from_finetuned(
    model_name_or_path,
    devices=devices,
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True)
embed_model.encode("hi")

logger.info(f"load embedding {os.path.basename(model_name_or_path)} model by {devices}.")
tokenized = AutoTokenizer.from_pretrained(model_name_or_path)
embed_model_dict[base_embed_model_name] = embed_model
embed_tokenizer_dict[base_embed_model_name] = tokenized

# 初始化rerank
rerank_model_dict, rerank_tokenizer_dict = {}, {}
# model_name_or_path = os.environ.get("RERANKER_PATH", "")
model_name_or_path = r'F:\inspur\EMBEDDING_MODEL\BAAI\bge-reranker-base'
if model_name_or_path == "":
    logger.info("Please set the environment variable 'RERANKER_PATH'")

base_rerank_model_name = os.path.basename(model_name_or_path)
logger.info(f"load rerank {base_rerank_model_name} model by {devices}.")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
reranker = FlagAutoReranker.from_finetuned(
    model_name_or_path,
    query_max_length=256,
    passage_max_length=512 if tokenizer.model_max_length > 100000000 else tokenizer.model_max_length,
    use_fp16=True,
    devices=devices)
reranker.compute_score(["hi", "hello"], normalize=True)
rerank_model_dict[base_rerank_model_name] = reranker
rerank_tokenizer_dict[base_rerank_model_name] = tokenizer


class RerankRequestModel(BaseModel):
    model: str = Field(base_rerank_model_name)
    query: str = Field("")
    documents: List = Field([""])
    normalize: bool = Field(True)
    cutoff_layers: int = Field(28)
    top_n: int = Field(default=0)
    return_documents: bool = Field(True)
    return_len: bool = Field(True)

class AddRerankRequestModel(BaseModel):
    model_path: str = Field("")
    model_name: str = Field("")
    model_type: str = Field(default="rerank")  # "rerank"

class DelRerankRequestModel(BaseModel):
    model_name: str = Field("")


class EmbeddingRequestModel(BaseModel):
    model: str = Field(base_embed_model_name)
    input: List = Field([""])


class EmbeddingSimilarityRequestModel(BaseModel):
    model: str = Field("")
    queries: List = Field([""])
    passages: List = Field([""])


class AddEmbeddingRequestModel(BaseModel):
    model_path: str = Field("")
    model_name: str = Field("")
    model_type: str = Field(default="embedd")  # "embedd/rerank"


class DelEmbeddingRequestModel(BaseModel):
    model_name: str = Field("")


def load_model(req, model_dict=None, tokenizer_dict=None):
    if model_dict is None:
        model_dict = {}
    if tokenizer_dict is None:
        tokenizer_dict = {}
    try:
        model_path = os.path.join(os.path.dirname(curr_path), "models", req.model_path)

        # 判断是embedd还是rerank
        if req.model_type == "embedd":
            if os.path.exists(model_path):
                logger.info(f"load model {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                model = FlagAutoModel.from_finetuned(
                    model_path,
                    query_max_length=256,
                    passage_max_length=512 if tokenizer.model_max_length > 10000000 else tokenizer.model_max_length,
                    use_fp16=True,
                    devices=devices)
            else:
                logger.info(f"load model {req.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(req.model_path)
                model = FlagAutoModel.from_finetuned(
                    req.model_path,
                    query_max_length=256,
                    passage_max_length=512 if tokenizer.model_max_length > 100000000 else tokenizer.model_max_length,
                    use_fp16=True,
                    devices=devices)

            model.encode("hi")

        else:
            if os.path.exists(model_path):
                logger.info(f"load model {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = FlagAutoReranker.from_finetuned(
                    model_path,
                    query_max_length=256,
                    passage_max_length=512 if tokenizer.model_max_length > 100000000 else tokenizer.model_max_length,
                    use_fp16=True,
                    devices=devices)
            else:
                logger.info(f"load model {req.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(req.model_path)
                model = FlagAutoReranker.from_finetuned(
                    req.model_path,
                    query_max_length=256,
                    passage_max_length=512 if tokenizer.model_max_length > 100000000 else tokenizer.model_max_length,
                    use_fp16=True,
                    devices=devices)

            model.compute_score(["hi", "hello"], normalize=True)

        if req.model_name != "":
            if req.model_name in model_dict:
                uid = "-" + str(uuid.uuid4())
                model_name = os.path.basename(req.model_name) + uid
                model_dict[model_name] = model
                tokenizer_dict[model_name] = tokenizer
                return True, model_name
            else:
                model_dict[os.path.basename(req.model_name)] = model
                tokenizer_dict[os.path.basename(req.model_name)] = tokenizer
                return True, os.path.basename(req.model_name)
        else:
            if os.path.basename(model_path) in model_dict:
                uid = "-" + str(uuid.uuid4())
                model_name = os.path.basename(model_path) + uid
                model_dict[model_name] = model
                tokenizer_dict[model_name] = tokenizer
                return True, model_name
            else:
                model_dict[os.path.basename(model_path)] = model
                tokenizer_dict[os.path.basename(model_path)] = tokenizer
                return True, os.path.basename(model_path)

    except Exception as e:
        logger.info(str(e))
        return False, ""


def del_model(model_name, model_dict=None, tokenizer_dict=None):
    try:
        if model_name == base_embed_model_name or model_name == base_rerank_model_name:
            logger.info(f"base model {model_name} is not allowed delete.")
            return False
        del model_dict[model_name]
        del tokenizer_dict[model_name]
        # torch.cuda.empty_cache()
        gc.collect()
        empty_cache()
        return True
    except Exception as e:
        logger.info(f"error del model {model_name}, {str(e)}")
        return False


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if is_xpu_available():
        torch.xpu.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()

def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False
