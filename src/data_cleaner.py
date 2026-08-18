import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    @staticmethod
    def clean_data(df, outlier_method='iqr'):
        df_clean = df.copy()
        logging.info("--- Валидация и очистка данных ---")
        
        # 1. Проверка и удаление дубликатов
        dup_count = df_clean.duplicated().sum()
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates()
            logging.info(f"Удалено дубликатов: {dup_count}")

        # 2. Преобразование строковых дат в datetime
        for col in df_clean.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                logging.info(f"Колонка '{col}' преобразована в datetime")

        # 3. Обработка пропущенных значений (замена медианой для чисел, модой для строк)
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    logging.info(f"Пропуски в '{col}' заполнены медианой: {fill_val}")
                else:
                    fill_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    logging.info(f"Пропуски в '{col}' заполнены модой: {fill_val}")

        # 4. Выявление и обработка выбросов (IQR или Z-score)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if 'id' in col.lower():
                continue
            
            if outlier_method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers > 0:
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    logging.info(f"Удалено выбросов (IQR) в '{col}': {outliers}")

            elif outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col]))
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    df_clean = df_clean[z_scores <= 3]
                    logging.info(f"Удалено выбросов (Z-score) в '{col}': {outliers}")

        # 5. Масштабирование числовых признаков
        if len(num_cols) > 0:
            cols_to_scale = [c for c in num_cols if 'id' not in c.lower()]
            if cols_to_scale:
                scaler = StandardScaler()
                df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])
                logging.info(f"Выполнено масштабирование (StandardScaler) для: {cols_to_scale}")

        # 6. Кодирование категориальных признаков (One-Hot Encoding)
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
            logging.info(f"Кодирование One-Hot Encoding выполнено для: {list(cat_cols)}")

        logging.info("Очистка данных успешно завершена.")
        return df_clean