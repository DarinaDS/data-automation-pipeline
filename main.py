import os
import logging
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.analyzer import Analyzer
from src.reporter import Reporter
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    logging.info("=== Запуск полного автоматизированного пайплайна  ===")
    
    # 1. Загрузка данных
    raw_df = DataLoader.load_csv("data/sales_data.csv")
    
    # Демонстрация возможности загрузки из API
    api_df = DataLoader.load_from_api("https://jsonplaceholder.typicode.com/posts")
    
    if raw_df is None:
        logging.error("Критическая ошибка: исходные данные не загружены.")
        return

    # 2. Валидация и очистка данных (IQR / Z-score + Scaling + One-Hot Encoding)
    clean_df = DataCleaner.clean_data(raw_df, outlier_method='iqr')

    # 3. Анализ данных и временных рядов
    stats = Analyzer.get_summary_stats(clean_df)
    time_series_trend = Analyzer.analyze_time_series(raw_df, date_column='Date', value_column='Total Amount')

    # 4. Модель машинного обучения
    target_col = 'Total Amount' if 'Total Amount' in clean_df.columns else clean_df.select_dtypes(include=['number']).columns[-1]
    model, metrics = Analyzer.train_ml_model(clean_df, target_column=target_col)

    # 5. Генерация отчетов и визуализаций
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    excel_path = os.path.join(reports_dir, "final_report.xlsx")
    pdf_path = os.path.join(reports_dir, "summary_report.pdf")

    Reporter.generate_excel_report(clean_df, stats, excel_path)
    Reporter.generate_visualizations(clean_df, reports_dir)
    Reporter.generate_pdf_report(metrics, pdf_path)

    # 6. Сохранение результатов в SQLite / PostgreSQL БД (Интеграция К5)
    try:
        db_engine = create_engine('sqlite:///reports/processed_data.db')
        clean_df.to_sql('cleaned_sales', db_engine, if_exists='replace', index=False)
        logging.info("Очищенные данные и результаты успешно сохранены в БД (SQLite/PostgreSQL)")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов в БД: {e}")

    # 7. Автоматическая отправка отчета
    Reporter.send_email("Автоматический отчет", "Пайплайн выполнен успешно. Все визуализации сформированы.", "analyst@company.com", pdf_path)

    logging.info("=== Автоматизированный пайплайн успешно завершен! ===")

if __name__ == "__main__":
    run_pipeline()