import os
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi import FastAPI, UploadFile, File, Form, status, Body
from tools.error_define import BinaryDecodingError
from server.router import utils_router, ResponseModel

'''上传文件'''
@utils_router.post(path="/upload_file", summary="bytes", response_model=ResponseModel, tags=["上传文件"])
async def upload_file(
        file: UploadFile = File(description="一个二进制文件"),
):
    # 验证文件
    pass

    # 将文件保存
    saved_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{file.filename}"
    try:
        file_content = await file.read()  # 读取上传文件的内容
        if os.path.exists(saved_path) and os.path.getsize(saved_path) == len(file_content):
            file_status = f"文件 {file.filename} 已存在。"
            content = {"isSuc": True, "code": 0, "msg": file_status, "res": {}}
            return JSONResponse(status_code=status.HTTP_200_OK, content=content)
        with open(saved_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        raise BinaryDecodingError(e)
    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": f""}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)
