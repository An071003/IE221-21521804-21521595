"""
    Đây là module dùng để import đầy đủ các thư viện và tạo ra các class hàm cần thiết cho cả project
"""
#Thư viện để xử lý dữ liệu
import pandas as pd
import numpy as np

#Thư viện để thống kê dữ liệu
from collections import Counter

#Thư viện để token hóa dữ liệu chữ
import spacy

#Thư viện để xử lý các vấn đề của học máy và xây dựng mô hình
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional

#Thư viện để trực quan hóa dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns


class Vectorize(TextVectorization):
    """
        Class vector hoá dữ liệu dạng text
    """

    def __init__(self, MAX_FEATURES=20000, output_sequence_length=1800, output_mode='int'):
        # Khởi tạo tham số
        super().__init__(max_tokens=MAX_FEATURES,
                         output_sequence_length=output_sequence_length,
                         output_mode=output_mode)

    def vectorizer_adapt(self, text):
        # Thích nghi với dữ liệu truyền vào
        return super().adapt(text)

    def call(self, text):
        # Vector hóa dữ liệu truyền vào
        return super().call(text)