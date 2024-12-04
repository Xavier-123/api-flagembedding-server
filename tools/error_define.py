

class CustomError(Exception):
    """自定义错误类型"""
    def __init__(self, message=""):
        super().__init__()
        self.message = message
        self.code = -1

    def __str__(self):
        return self.message

class BinaryDecodingError(CustomError):
    """二进制文件解码错误"""

    def __init__(self, message=""):
        super().__init__()
        self.message = f"Binary decoding error! {message}"
        self.code = 1