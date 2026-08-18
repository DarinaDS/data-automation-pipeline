import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

class Reporter:
    @staticmethod
    def generate_excel_report(df, stats, file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            if stats is not None:
                stats.to_excel(writer, sheet_name='Stats')
        logging.info(f"Excel отчет сохранен: {file_path}")

    @staticmethod
    def generate_visualizations(df, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Matplotlib + Seaborn
        plt.figure(figsize=(10, 6))
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            sns.histplot(df[num_cols[0]], kde=True, color='purple')
            plt.title(f'Распределение и плотность: {num_cols[0]}')
            chart_path = os.path.join(output_dir, "seaborn_chart.png")
            plt.savefig(chart_path)
            plt.close()
            logging.info(f"График Seaborn сохранен: {chart_path}")

        # 2. Интерактивный график Plotly (сохранение в HTML)
        if len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title="Интерактивный анализ зависимостей")
            plotly_path = os.path.join(output_dir, "interactive_chart.html")
            fig.write_html(plotly_path)
            logging.info(f"Интерактивный график Plotly сохранен: {plotly_path}")

    @staticmethod
    def generate_pdf_report(metrics, file_path):
        c = canvas.Canvas(file_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Отчет по автоматизации обработки данных")
        c.drawString(100, 740, "--------------------------------------------------------")
        
        c.setFont("Helvetica", 12)
        y = 700
        c.drawString(100, y, "Ключевые метрики машинного обучения (ML):")
        y -= 25
        for metric_name, value in metrics.items():
            c.drawString(120, y, f"{metric_name}: {value:.4f}")
            y -= 20
            
        c.save()
        logging.info(f"PDF отчет сохранен: {file_path}")

    @staticmethod
    def send_email(subject, body, to_email, attachment_path=None):
        try:
            msg = MIMEMultipart()
            msg['From'] = "automation@company.com"
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(attachment_path)}")
                msg.attach(part)

            logging.info(f"Модуль SMTP Email-рассылки инициализирован для адреса: {to_email}")
        except Exception as e:
            logging.error(f"Ошибка в модуле отправки Email: {e}")