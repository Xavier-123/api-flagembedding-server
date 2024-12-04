import json
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, Form, status, Body
from FlagEmbedding.inference.embedder.model_mapping import AUTO_EMBEDDER_MAPPING

from tools.log import logger
from tools.utils import EmbeddingRequestModel, EmbeddingSimilarityRequestModel, AddEmbeddingRequestModel, \
    DelEmbeddingRequestModel, load_model, del_model, embed_model_dict, embed_tokenizer_dict
from tools.embedding_utils import embed_infer, similarity_comparison
from server.router import embed_router, ResponseModel



@embed_router.post(path="/v1/embeddings", summary="embedding", response_model=ResponseModel,
                   tags=["向量化"])
async def embeddings(
        req: EmbeddingRequestModel,
):
    try:
        content = embed_infer(req, embed_model_dict, embed_tokenizer_dict)
        logger.info(f">>> content: {content}")
    except Exception as e:
        content = {"isSuc": False, "code": 0, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")
    if isinstance(content, dict):
        if "msg" in content:
            content["msg"] = str(content["msg"])
        return JSONResponse(status_code=status.HTTP_200_OK, content=content)
    return JSONResponse(status_code=status.HTTP_200_OK, content=json.dumps(content))


@embed_router.post(path="/v1/similarity", summary="similarity", response_model=ResponseModel,
                   tags=["计算相似度"])
async def similarity(
        req: EmbeddingSimilarityRequestModel,
):
    try:
        content = similarity_comparison(req, embed_model_dict, embed_tokenizer_dict)
    except Exception as e:
        content = {"isSuc": True, "code": 0, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


# 更改embedding模型
@embed_router.post(path="/v1/add_emmbedding", summary="添加 embedding 模型", response_model=ResponseModel,
                   tags=["添加 embedding 模型"])
async def add_emmbedding(
        req: AddEmbeddingRequestModel,
):
    '''添加embedding模型'''
    try:
        isSucess, model_name = load_model(req, embed_model_dict, embed_tokenizer_dict)
        if isSucess:
            logger.info(f">>>Success add embed model {model_name}")
            content = {"isSuc": True, "code": 0, "msg": f"Success add embed model {model_name}",
                       "res": {"model_name": model_name}}
        else:
            logger.info(f">>>Failed add embed model {req.model_path}")
            content = {"isSuc": False, "code": -1, "msg": f"Failed add embed model {req.model_path}", "res": {}}
    except Exception as e:
        logger.info(f">>> error:{str(e)}")
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


# 删除embedding模型
@embed_router.post(path="/v1/delete_embedding", summary="删除 embedding 模型", response_model=ResponseModel,
                   tags=["删除 embedding 模型"])
async def delete_embedding(
        req: DelEmbeddingRequestModel,
):
    '''删除 embedding 模型'''
    try:
        if req.model_name != "":
            isSucess = del_model(req.model_name, embed_model_dict, embed_tokenizer_dict)
            if isSucess:
                content = {"isSuc": True, "code": 0, "msg": f"del embed model {req.model_name}", "res": {}}
            else:
                content = {"isSuc": False, "code": -1, "msg": f"failed del embed model {req.model_name}", "res": {}}
        else:
            logger.info(f">>> model_name is Null。")
            content = {"isSuc": False, "code": -1, "msg": f">>> model_name is Null。", "res": {}}

        logger.info(f">>>Del embed model {req.model_name}")
    except Exception as e:
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@embed_router.get(path="/v1/embeddings", summary="查询 embedding 模型", response_model=ResponseModel,
                  tags=["查询 embedding 模型"])
async def get_embeddings():
    '''查询 embedding 模型'''
    try:
        res = {
            "Running model": list(embed_model_dict.keys()),
            "Supported models": list(AUTO_EMBEDDER_MAPPING.keys())
        }
        content = {"isSuc": True, "code": 0, "msg": "", "res": res}
    except Exception as e:
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


if __name__ == '__main__':
    sentences_1 = ["I love NLP", "I love machine learning"]
    sentences_2 = ["I love BGE", "I love text retrieval"]
    # embeddings_1 = model.encode(sentences_1)
    # embeddings_2 = model.encode(sentences_2)
    # similarity = embeddings_1 @ embeddings_2.T
    # print(similarity)
    # # print(embeddings_1)
    #
    # queries = ['query_1', 'query_2']
    # passages = ["样例文档-1", "样例文档-2"]
    # q_embeddings = model.encode_queries(queries)
    # p_embeddings = model.encode_corpus(passages)
    # scores = q_embeddings @ p_embeddings.T
    # print(scores)
    # print("1111111111")
