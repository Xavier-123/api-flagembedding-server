import os
from tools.utils import devices, EmbeddingRequestModel
from tools.log import logger



def embed_infer(req: EmbeddingRequestModel, embed_model_dict, embed_tokenized_dict):
    try:
        model = embed_model_dict.get(req.model, -1)
        tokenizer = embed_tokenized_dict.get(req.model, -1)
        if model == -1 or tokenizer == -1:
            return {"isSuc": False, "code": 0, "msg": f"model {req.model} is not exist", "res": {}}

        if "bge-m3" == req.model:
            embeddings = model.encode(req.inputs)["dense_vecs"].tolist()
            logger.info(embeddings)
        else:
            embeddings = model.encode(req.inputs).tolist()

        data = []
        for idx, embedding in enumerate(embeddings):
            embed = {}
            embed["index"] = idx
            embed["object"] = "embedding"
            embed["embedding"] = embedding
            embed["usage"] = {"prompt_tokens": len(tokenizer(req.inputs[idx])['input_ids']),
                              "total_tokens": len(tokenizer(req.inputs[idx])['input_ids'])}
            data.append(embed)
        content = {"object": "list", "model": req.model, "data": data}
        return content
    except Exception as e:
        return {"isSuc": False, "code": 0, "msg": e, "res": {}}



def similarity_comparison(req: EmbeddingRequestModel, embed_model_dict, embed_tokenized_dict):
    try:
        model = embed_model_dict.get(req.model, -1)
        tokenizer = embed_tokenized_dict.get(req.model, -1)
        if model == -1 or tokenizer == -1:
            return {"isSuc": False, "code": 0, "msg": f"model {req.model} is not exist", "res": {}}

        q_embeddings = model.encode_queries(req.queries)
        p_embeddings = model.encode_corpus(req.passages)
        scores = q_embeddings @ p_embeddings.T
        content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": {"scores": scores.tolist()}}
        return content
    except Exception as e:
        return {"isSuc": False, "code": 0, "msg": e, "res": {}}


