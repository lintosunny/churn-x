import os 
import sys 
import yaml
import dill
import json
import numpy as np 
from ml.logger import logging
from ml.exception import TelcoChurnMLException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file) 
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e
    

def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e
    

def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e
    

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e
    

def read_json_file(file_path: str) -> dict:
    """
    Reads a JSON file and returns its content as a Python dictionary.
    """
    try:
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e
    

def write_json_file(file_path: str, content: object, replace: bool = True) -> None:
    """
    Writes a Python object (dict, list, etc.) to a JSON file.
    If replace=True and the file exists, it will be overwritten.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as json_file:
            json.dump(content, json_file, indent=4)  # pretty print
    except Exception as e:
        raise TelcoChurnMLException(e, sys) from e