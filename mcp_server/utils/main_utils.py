import os 
import sys 
import dill
from mcp_server.logger import logging
from mcp_server.exception import TelcoChurnMCPException
    

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise TelcoChurnMCPException(e, sys) from e