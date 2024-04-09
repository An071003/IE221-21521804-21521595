# IE221-21521804-21521959

Trường Đại học Công nghệ Thông tin - UIT  
Đồ án môn kỹ thuật lập trình Python - IE221.O23.CNCL  


**#Hồ Vũ An - 21521804**
**#Nguyễn Thành Trung - 21521595**

# Giới thiệu:  
Đồ án này nghiên cứu các thư viện liên quan đến học máy và ngôn ngữ tự nhiên và áp dụng chúng để tạo ra một ứng dụng desktop có thể cho phép người dùng nhập vào một đoạn bình luận, đưa ra kết quả dư đoán của mô hình học máy là bình luận tiêu cực hoặc không tiêu cực.  

## Nhiệm vụ tuần 1: Nghiên cứu các thư viện học máy và xử lý ngôn ngữ tự nhiên.  
### Giới thiệu về thư viện scikit-learn  
- **Scikit-learn** là thư viện python cung cấp nhiều công cụ và thuật toán để thực hiện nhiều nhiệm vụ khác nhau liên quan đến việc xây dựng và huấn luyện mô hình máy học, bao gồm phân loại, hồi quy, phân cụm và giảm chiều và xử lý dữ liệu.  
### Giới thiệu về thư viện tensonflow và keras   
- **Tensoflow** cung cấp một cách linh hoạt để xây dựng và huấn luyện các mô hình học máy trên các máy tính cá nhân và các hệ thống phân tán lớn. Hơn nữa, **TensorFlow** cho phép xây dựng các mô hình neural network phức tạp, dự đoán và phân loại.  
- **Keras** được sử dụng phổ biến trong việc phát triển các mô hình xử lý ngôn ngữ tự nhiên (NLP). Các khung làm việc và công cụ mạnh mẽ cho việc xây dựng và huấn luyện các mô hình NLP như RNN, LSTM, GRU, CNN và các kiến trúc thay thế như BERT và GPT được cung cấp bởi thư viện này. Ngoài ra, các mô-đun tiêu chuẩn, layer và thuật toán tối ưu hóa đều được cung cấp bởi **Keras**.  
### Giới thiệu về thư viện spacy
- **Spacy** là một thư viện Python được thiết kế để thực hiện xử lý ngôn ngữ tự nhiên (NLP). Nó cung cấp các công cụ mạnh mẽ cho việc xử lý và phân tích văn bản, chẳng hạn như tách từ , phân loại từ loại, phân tích cú pháp, phân tích định danh, chuyển đổi văn bản thành vector và loại bỏ stop words và nhiều chức năng khác.  
## Ứng dụng các thư viện trong đồ án này:  
### Spacy và scikit-learn  
- Sử dụng **Spacy** để loại bỏ dấu và loại bỏ các từ stop words để xử lý dữ liệu.  
- Chuyển đổi dữ liệu văn bản thành ma trận vector.  
- Sử dụng hàm train_test_split trong thư viện **Scikit-learn** để chi dữ liệu thành X_train, X_test, Y_train, Y_test.  
- Sử dụng mô hình KNeighborsClassifier trong thư viện **Scikit-learn** để đưa ra dự đoán.  
- Sử dụng hàm classification_report để đưa ra các thông số của mô hình như Accuracy, Precision, Recall, F1-score.
