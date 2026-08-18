import pandas as pd
import requests
import logging
from sqlalchemy import create_engine

class DataLoader:
    @staticmethod
    def load_csv(file_path):
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Успешно загружен CSV: {file_path}")
            return df
        except Exception as e:
            logging.error(f"Ошибка загрузки CSV: {e}")
            return None

    @staticmethod
    def load_excel(file_path):
        try:
            df = pd.read_excel(file_path)
            logging.info(f"Успешно загружен Excel: {file_path}")
            return df
        except Exception as e:
            logging.error(f"Ошибка загрузки Excel: {e}")
            return None

    @staticmethod
    def load_from_api(url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            logging.info(f"Успешно загружены данные из API: {url}")
            return df
        except Exception as e:
            logging.error(f"Ошибка загрузки из API: {e}")
            return None

    @staticmethod
    def load_from_db(connection_string, query):
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logging.info("Успешно выполнен SQL-запрос и загружены данные из БД")
            return df
        except Exception as e:
            logging.error(f"Ошибка загрузки из БД: {e}")
            return None