from etl.data_loader import TelcoDataExtract
from etl.constants import FILE_PATH, DATABASE, COLLECTION


telco = TelcoDataExtract()

if __name__ == '__main__':
    records = telco.csv_to_json_convertor(file_path=FILE_PATH)
    telco.insert_data_mongodb(records=records, database=DATABASE, collection=COLLECTION)