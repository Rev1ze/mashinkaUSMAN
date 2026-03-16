import sys
import re
import pickle

from functools import lru_cache

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from string import punctuation
import pymorphy2
import mysql.connector
from mysql.connector import Error

#пипец полный, как такое можно было сделать, автор быдло
#это же ужас, стыд и срам
#этому дураку больше не стоит выходить на улицуzz
#
# ---- preprocessing helpers (from notebook) ----

russian_stopwords = set(stopwords.words('russian'))
punctuation_set = set(punctuation + '—–…')

morph = pymorphy2.MorphAnalyzer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace('ё', 'е')
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', ' USER ', text)
    text = re.sub(r'#\w+', ' HASHTAG ', text)
    text = re.sub(r'[^а-яa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@lru_cache(maxsize=100000)
def lemmatize_word(word: str) -> str:
    p = morph.parse(word)
    if not p:
        return word
    return p[0].normal_form


def lemmatize_text(text: str) -> str:
    words = text.split()
    filtered = []
    for word in words:
        if word in russian_stopwords or word in punctuation_set or len(word) <= 2:
            continue
        filtered.append(lemmatize_word(word))
    return ' '.join(filtered)


def get_prediction_scores(input_text: str, model, tokenizer):
    clean = clean_text(input_text)
    lemm = lemmatize_text(clean)

    seq = tokenizer.texts_to_sequences([lemm])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

    scores = model.predict(padded, verbose=0)[0]
    return {
        'normal': float(scores[0]),
        'insult': float(scores[1]),
        'threat': float(scores[2]),
        'obscenity': float(scores[3]),
    }


def get_tone_label(scores: dict) -> str:
    # корректный выбор наиболее вероятного класса
    classes = ['normal', 'insult', 'threat', 'obscenity']
    best = max(classes, key=lambda c: scores[c])
    return best


class ToxicityCheckerApp(QMainWindow):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'root',
            'database': 'toxicity_db'
        }
        self.db_connection = None
        self.db_cursor = None

        self.setWindowTitle('Toxicity Analyzer - Official Edition')
        self.setMinimumSize(720, 520)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.apply_style()
        self.setup_ui()

        self.connect_db()

    def apply_style(self):
        self.setStyleSheet(
            """
            QWidget { background-color: #f5f9ff; color: #1f3a72; }
            QLabel#title { font-size: 22px; font-weight: bold; color: #1b4f88; }
            QTextEdit, QLineEdit { background-color: white; border: 1px solid #8ab4ff; border-radius: 6px; }
            QPushButton { background-color: #4d86ff; color: white; border-radius: 6px; padding: 8px; }
            QPushButton:hover { background-color: #2f6dff; }
            QPushButton:pressed { background-color: #1f55e0; }
            """
        )

    def setup_ui(self):
        layout = QVBoxLayout()

        title_label = QLabel('Toxicity Analyzer (Python + PyQt5)')
        title_label.setObjectName('title')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 18, QFont.Bold))
        layout.addWidget(title_label)

        info = QLabel('Введите сообщение в поле ниже, затем нажмите "Анализировать"')
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText('Введите текст (русский)')
        self.input_text.setMinimumHeight(160)
        layout.addWidget(self.input_text)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        analyze_button = QPushButton('Анализировать')
        analyze_button.clicked.connect(self.run_analysis)
        button_layout.addWidget(analyze_button)

        clear_button = QPushButton('Очистить')
        clear_button.clicked.connect(self.input_text.clear)
        button_layout.addWidget(clear_button)

        layout.addLayout(button_layout)

        self.result_label = QLabel('Результат: нет данных')
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.scores_label = QLabel('Оценки: —')
        self.scores_label.setWordWrap(True)
        layout.addWidget(self.scores_label)

        self.central_widget.setLayout(layout)

    def connect_db(self):
        try:
            self.db_connection = mysql.connector.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
            )
            self.db_cursor = self.db_connection.cursor()
            self.db_cursor.execute('CREATE DATABASE IF NOT EXISTS `%s`' % self.db_config['database'])
            self.db_connection.database = self.db_config['database']
            self.create_table()
        except Error as e:
            QMessageBox.warning(self, 'БД не подключена', f'Не удалось подключиться к MySQL: {e}')
            self.db_connection = None
            self.db_cursor = None

    def create_table(self):
        if self.db_cursor is None:
            return
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text_input TEXT,
                result_label VARCHAR(32),
                score_normal FLOAT,
                score_insult FLOAT,
                score_threat FLOAT,
                score_obscenity FLOAT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''
        self.db_cursor.execute(create_table_query)
        self.db_connection.commit()

    def save_analysis(self, original_text, tone, scores):
        if self.db_connection is None or self.db_cursor is None:
            return
        try:
            insert_query = '''
                INSERT INTO analysis_logs
                (text_input, result_label, score_normal, score_insult, score_threat, score_obscenity)
                VALUES (%s, %s, %s, %s, %s, %s)
            '''
            self.db_cursor.execute(insert_query, (
                original_text,
                tone,
                scores['normal'],
                scores['insult'],
                scores['threat'],
                scores['obscenity']
            ))
            self.db_connection.commit()
        except Error as e:
            QMessageBox.warning(self, 'Ошибка БД', f'Не удалось сохранить результат в БД: {e}')

    def run_analysis(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, 'Ошибка', 'Пожалуйста, введите текст для анализа.')
            return

        try:
            scores = get_prediction_scores(text, self.model, self.tokenizer)
            tone = get_tone_label(scores)
            self.result_label.setText(f'Результат: {tone.upper()}')
            self.scores_label.setText(
                'Оценки:\n'
                f'Нормальное сообщение: {scores["normal"]:.4f}\n'
                f'Оскорбление: {scores["insult"]:.4f}\n'
                f'Угроза: {scores["threat"]:.4f}\n'
                f'Непристойность: {scores["obscenity"]:.4f}'
            )
            self.save_analysis(text, tone, scores)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось выполнить анализ:\n{e}')


def load_resources(model_path='toxic_model.h5', tokenizer_path='tokenizer.pickle'):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def main():
    app = QApplication(sys.argv)

    try:
        model, tokenizer = load_resources()
    except Exception as ex:
        QMessageBox.critical(None, 'Ошибка загрузки', f'Не удалось загрузить модель/токенайзер: {ex}')
        sys.exit(1)

    window = ToxicityCheckerApp(model, tokenizer)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
