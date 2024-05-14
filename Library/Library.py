"""
    Đây là module dùng để import đầy đủ các thư viện và tạo ra các class hàm cần thiết cho cả project
"""
# Thư viện để xử lý dữ liệu
import pandas as pd
import numpy as np

# Thư viện để thống kê dữ liệu
from collections import Counter

# Thư viện để token hóa dữ liệu chữ
import spacy

# Thư viện để xử lý các vấn đề của học máy và xây dựng mô hình
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional
import keras

# Thư viện để trực quan hóa dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

# Thư viện lưu và tải mô hình TextVectorization
import pickle as pkl

# Thư viện tạo ra app_desktop
import tkinter as tk
from tkinter import messagebox

class Vectorize(TextVectorization):
    """
        Class vector hoá dữ liệu dạng text
    """
    def __init__(self, MAX_FEATURES=20000, output_sequence_length=1800, output_mode='int', vocabulary=None):
        """
        Khởi tạo thông số cho class
        :param MAX_FEATURES: Số lượng từ tối đa trong từ vựng
        :param output_sequence_length: Độ dài mà mỗi chuỗi văn bản
        :param output_mode: Dạng đầu ra
        :param vocabulary: Từ vựng vector hóa
        """
        super().__init__(max_tokens=MAX_FEATURES,
                         output_sequence_length=output_sequence_length,
                         output_mode=output_mode,
                         vocabulary=vocabulary)

    def vectorizer_adapt(self, text):
        """
        Hàm thích nghi lớp với dữ liệu được truyền vào
        :param text: dữ liệu văn bản
        :return: lớp vector đã được thích nghi
        """
        # Thích nghi với dữ liệu truyền vào
        return super().adapt(text)

    def __call__(self, text):
        """
        Hàm chuyển văn bản thành vector
        :param text: văn bản cần chuyển
        :return: vector đã chuyển đổi
        """
        return super().__call__(text)

    def save_vocabulary(self, path):
        """
        Hàm lưu từ vựng lớp TextVectorization
        :param path: đường dẫn lưu
        """
        with open(path, 'wb') as f:
            pkl.dump(self.get_vocabulary(), f)