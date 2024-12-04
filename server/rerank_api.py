from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, Form, status, Body
from FlagEmbedding.inference.reranker.model_mapping import AUTO_RERANKER_MAPPING

from tools.log import logger
from tools.utils import RerankRequestModel, AddRerankRequestModel, DelRerankRequestModel, load_model, del_model,\
    rerank_model_dict, \
    rerank_tokenizer_dict
from tools.rerank_utils import rerank_infer
from server.router import embed_router, ResponseModel


@embed_router.post(path="/v1/rerank", summary="rerank", response_model=ResponseModel,
                   tags=["重排序"])
async def rerank(
        req: RerankRequestModel,
):
    try:
        content = rerank_infer(req, rerank_model_dict)
        logger.info(f">>> content: {content}")
    except Exception as e:
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@embed_router.post(path="/v1/add_rerank", summary="添加rerank模型", response_model=ResponseModel,
                   tags=["添加rerank模型"])
async def add_rerank(
        req: AddRerankRequestModel,
):
    '''添加 rerank 模型'''
    try:
        isSucess, model_name = load_model(req, rerank_model_dict, rerank_tokenizer_dict)
        if isSucess:
            logger.info(f">>>Success add rerank model {model_name}")
            content = {"isSuc": True, "code": 0, "msg": f"Success add rerank model {model_name}",
                       "res": {"model_name": model_name}}
        else:
            logger.info(f">>>Failed add embed model {req.model_path}")
            content = {"isSuc": False, "code": -1, "msg": f"Failed add rerank model {req.model_path}", "res": {}}
    except Exception as e:
        logger.info(f">>> error:{str(e)}")
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@embed_router.post(path="/v1/delete_rerank", summary="删除rerank模型", response_model=ResponseModel,
                   tags=["删除rerank模型"])
async def delete_rerank(
        req: DelRerankRequestModel,
):
    '''删除 rerank 模型'''
    try:
        if req.model_name != "":
            isSucess = del_model(req.model_name, rerank_model_dict, rerank_tokenizer_dict)
            if isSucess:
                content = {"isSuc": True, "code": 0, "msg": f"del rerank model {req.model_name}", "res": {}}
            else:
                content = {"isSuc": False, "code": -1, "msg": f"failed del rerank model {req.model_name}", "res": {}}
        else:
            logger.info(f">>> model_name is Null。")
            content = {"isSuc": False, "code": -1, "msg": f">>> model_name is Null。", "res": {}}

        logger.info(f">>>Del embed model {req.model_name}")
    except Exception as e:
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@embed_router.get(path="/v1/rerank", summary="查询 rerank 模型", response_model=ResponseModel,
                   tags=["查询 rerank 模型"])
async def get_rerank():
    '''查询 rerank 模型'''
    try:
        res = {
            "Running model": list(rerank_model_dict.keys()),
            "Supported models": list(AUTO_RERANKER_MAPPING.keys())
        }
        content = {"isSuc": True, "code": 0, "msg": "", "res": res}
    except Exception as e:
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}
        logger.info(f">>> error:{str(e)}")

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


if __name__ == '__main__':
    pass
