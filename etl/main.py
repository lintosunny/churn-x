from etl.load import TelcoDataLoad
from etl.transform import TelcoDataTransform
from etl.constants import RAW_FILE_PATH, TRANSFORMED_FILE_PATH, DATABASE, COLLECTION

data_transformer = TelcoDataTransform()
data_loader = TelcoDataLoad()

if __name__ == '__main__':
    data_transformer.initiate_data_transformation()
    records = data_loader.csv_to_json_convertor(file_path=TRANSFORMED_FILE_PATH)
    data_loader.insert_data_mongodb(records=records, database=DATABASE, collection=COLLECTION)