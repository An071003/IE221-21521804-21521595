# Import Library
import Library.Library as Lb
# Import Data
import Data.Data as Dt

class MyModel(Lb.Sequential):
    """
       Class Model
    """
    def __init__(self, name=None):
        """
           khởi tạo giá trị cho class
        """
        super(MyModel, self).__init__(name=name)
        # Lớp Embedding với số lượng từ vựng tối đa là 20000 từ đã xác định + 1 là những từ chưa xác định
        self.add(Lb.Embedding(20000 + 1, 32))
        # Lớp Bidirectional để huấn luyện hai lớp LSTM theo 2 hướng thuận và ngược
        self.add(Lb.Bidirectional(Lb.LSTM(32, activation='tanh')))
        # 3 lớp Dense được sử dụng để tạo ra các đặc trưng
        self.add(Lb.Dense(128, activation='relu'))
        self.add(Lb.Dense(256, activation='relu'))
        self.add(Lb.Dense(128, activation='relu'))
        # Lớp Dropout để tránh overfitting
        self.add(Lb.Dropout(0.5))
        # Lớp cuối cùng có 6 nơ-ron tương ứng 6 cột taget_column với hàm kich hoạt  là sigmoid
        self.add(Lb.Dense(6, activation='sigmoid', name="output_layer"))


    def compile(self, loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall']):
        """
        Hàm compile mô hình
        @param loss: 'binary_crossentropy'
        @param optimizer: 'adam'
        @param metrics:'accuracy', 'precision', 'recall'
        """
        super().compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train_model(self, train, val, epochs=5):
        """
        Hàm huấn luyện mô hình
        :param train: Tập dữ liệu train
        :param val: Tập dữ liệu val
        :param epochs: số vòng lập huấn luyện mô hình
        :return: mô hình đã train
        """
        return super().fit(train, epochs=epochs, validation_data=val)

    def visualisation_train_model(self, History):
        """
        Hàm Trực quan hóa số liêu mô hình sau khi huấn luyện
        :param History: dữ liệu mô hình đã huấn luyện
        """
        Lb.plt.figure(figsize=(20, 10))
        Lb.pd.DataFrame(History.history).plot()
        Lb.plt.title('History')
        Lb.plt.xlabel('Epoch')
        Lb.plt.ylabel('percent')
        Lb.plt.legend(loc='lower right')
        Lb.plt.show()

    def predict(self, vectorizer, text):
        """
        Hàm tạo dữ đoán với câu mẫu
        :param vectorizer: lớp vector hóa dữ liệu dạng text
        :param text: câu bình luận
        :return: Kết quả dự đoán
        """
        check = vectorizer(text)
        return super().predict(Lb.np.expand_dims(check, 0))

    def visualisation_prediction(self, target_columns, prediction):
        """
        Hàm trực quan hóa kết quả dự đoán
        :param target_columns: tên các mục dự đoán
        :param prediction: kết quả dự đoán
        """
        # Tạo đồ thị
        Lb.plt.figure(figsize=(10, 6))
        Lb.sns.barplot(x=target_columns, y=prediction, palette="viridis")
        Lb.plt.title('prediction')
        Lb.plt.ylabel('precent')
        Lb.plt.xlabel('Labels')
        Lb.plt.show()

    def evaluate_model(self, test):
        """
        Hàm tạo đánh giá mô hình
        :param test: dữ liệu test
        :return: kết quả đánh giá
        """
        return super().evaluate(test)

    def visualisation_evalution(self, evalution):
        """
        Hàm trực quan hóa đánh giá
        :param evalution: kết quả đánh giá
        """
        metrics = ['loss', 'acc', 'precision', 'recall']  # Các thông số đánh giá
        # Tạo đồ thị
        Lb.plt.figure(figsize=(10, 6))
        Lb.sns.barplot(x=metrics, y=evalution, palette="viridis")
        Lb.plt.title('Performance Metrics')
        Lb.plt.ylabel('percent')
        Lb.plt.xlabel('Metrics')
        Lb.plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
        Lb.plt.show()