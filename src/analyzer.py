import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Analyzer:
    @staticmethod
    def get_summary_stats(df):
        stats_df = df.describe(include='all').T
        stats_df['median'] = df.median(numeric_only=True)
        stats_df['mode'] = df.mode(numeric_only=True).iloc[0] if not df.mode(numeric_only=True).empty else np.nan
        logging.info("Базовые статистики (среднее, медиана, мода, стд. откл.) рассчитаны.")
        return stats_df

    @staticmethod
    def analyze_time_series(df, date_column, value_column):
        try:
            if date_column not in df.columns:
                logging.warning(f"Колонка даты {date_column} не найдена для анализа временного ряда.")
                return None
            
            ts_df = df.copy()
            ts_df[date_column] = pd.to_datetime(ts_df[date_column])
            ts_df = ts_df.set_index(date_column).sort_index()
            
            # Ресемплирование по месяцам для выявления тренда и сезонности
            monthly_trend = ts_df[value_column].resample('ME').mean()
            logging.info("Анализ временных рядов (тренд/сезонность по месяцам) успешно выполнен.")
            return monthly_trend
        except Exception as e:
            logging.error(f"Ошибка при анализе временного ряда: {e}")
            return None

    @staticmethod
    def train_ml_model(df, target_column):
        try:
            if target_column not in df.columns:
                logging.warning(f"Целевая колонка {target_column} не найдена для ML.")
                return None, {}

            X = df.drop(columns=[target_column])
            # Оставляем только числовые и bool столбцы
            X = X.select_dtypes(include=['number', 'bool'])
            y = df[target_column]

            if X.empty:
                logging.warning("Нет подходящих признаков для обучения модели.")
                return None, {}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            logging.info(f"ML модель обучена. Метрики: {metrics}")
            return model, metrics
        except Exception as e:
            logging.error(f"Ошибка обучения ML-модели: {e}")
            return None, {}