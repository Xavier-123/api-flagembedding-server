from fastapi import APIRouter
from pydantic import BaseModel, Field


class ResponseModel(BaseModel):
    isSuc: bool = Field(True, example=True)
    code: int = Field(0, example=0)
    msg: str = Field("Succeed~", example="Succeed~")
    res: dict = Field({}, example={})


# 路由
utils_router = APIRouter()
embed_router = APIRouter()
rerank_router = APIRouter()

# utils_api
from server.utils_api import upload_file

# embed_api
from server.embedding_api import embeddings, similarity, get_embeddings
# from server.embedding_api import add_emmbedding, delete_embedding

# rerank_api
from server.rerank_api import rerank, get_rerank
# from server.rerank_api import add_rerank, delete_rerank
