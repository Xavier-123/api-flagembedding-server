from tools.utils import devices, RerankRequestModel, rerank_tokenizer_dict
from tools.log import logger


def rerank_infer(req: RerankRequestModel, rerank_model_dict=None):
    try:
        kwargs = {}
        model = rerank_model_dict.get(req.model, -1)
        tokenizer = rerank_tokenizer_dict.get(req.model, -1)
        if model == -1 or tokenizer == -1:
            return {"isSuc": False, "code": 0, "msg": f"model {req.model} is not exist", "res": {}}

        corpus = []

        input_tokens = 0
        for document in req.documents:
            input_tokens += len(tokenizer(document)['input_ids'])
            corpus.append([req.query, document])
        if not req.return_len:
            input_tokens = None

        logger.info(f"the tokenizer class is {type(tokenizer).__name__}")

        # 分数归一标准化
        if req.normalize:
            kwargs["normalize"] = req.normalize
        if type(tokenizer).__name__ == "LlamaTokenizerFast":
            kwargs["cutoff_layers"] = [req.cutoff_layers]

        scores = model.compute_score(corpus, **kwargs)


        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        if len(corpus) > req.top_n:
            sorted_indices = sorted_indices[:req.top_n]

        data = []
        for idx in sorted_indices:
            embed = {}
            embed["index"] = idx
            embed["object"] = "rerank"
            embed["relevance_score"] = scores[idx] if str(scores[idx]) != 'nan' else 'nan'
            # embed["document"]["text"] = corpus[idx]
            embed["document"] = {"text": corpus[idx][1] if req.return_documents else None}
            data.append(embed)
        content = {
            "object": "list",
            "model": req.model,
            "data": data,
            # "usage": {"total_tokens": None, "prompt_tokens": None}
            "meta": {
                "api_version": None,
                "billed_units": None,
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": input_tokens
                },
                "warnings": None
            }
        }
        return content
    except Exception as e:
        return {"isSuc": False, "code": -1, "msg": str(e), "res": {}}


if __name__ == '__main__':
    pass
