# IE221-21521804-21521959

Trường Đại học Công nghệ Thông tin - UIT  
Đồ án môn kỹ thuật lập trình Python - IE221.O23.CNCL  


**#21521804 - Hồ Vũ An**  
**#21521595 - Nguyễn Thành Trung**

# Giới thiệu:  
Đồ án này nghiên cứu các thư viện liên quan đến học máy và ngôn ngữ tự nhiên và áp dụng chúng để tạo ra một ứng dụng desktop có thể cho phép người dùng nhập vào một đoạn bình luận, đưa ra kết quả dư đoán của mô hình học máy là bình luận tiêu cực hoặc không tiêu cực.  

# Dataset:  
− Tập dữ liệu mà nhóm chúng em quyết định sử dụng được lấy từ một số lượng lớn các bình luận trên Wikipedia đã dược đánh giá là có hành vi tiêu cực. Các loại tiêu cực đó là:  
 + toxic  
 + severe_toxic  
 + obscene  
 + threat  
 + insult  
 + identity_hate  

− Mô tả tập tin csv: tập tin mà bọn em sẽ cung cấp để huấn luyện mô hình học máy train.csv (tập huấn luyện chứa các bình luận có nhãn nhị phân)  
link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

# Các Thư viện sử dụng:  
- Thư viện python:
```python
  !pip install tensorflow 
  !pip install tensorflow-gpu 
  !pip install keras 
  !pip install pandas 
  !pip install seaborn 
  !pip install matplotlib 
  !pip install numpy 
  !pip install spacy 
```  
- Pagekage spacy english pipeline:  
```python
  !python -m spacy download en_core_web_lg
```
