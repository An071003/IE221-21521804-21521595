# import thư viện
import Library.Library as Lb
import Model.Model as Md
import App_desktop.App_desktop as Ap

if __name__ == '__main__':
    # Tải mô hình
    model = Md.MyModel()
    model.build((None, 1800))
    model.load_weights('./Save_model/model.weights.h5')

    # Cài đặt lớp vector hóa từ vocabulary đã lưu
    with open('./Save_model/vectorizer.pkl', 'rb') as f:
        vocabulary = Lb.pkl.load(f)

    vectorizer = Lb.Vectorize(vocabulary=vocabulary)

    # Các nhãn dự đoán
    target_columns = list(Lb.np.array(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))

    # Cài đặt app desktop
    app = Ap.ToxicityPredictorApp(model, vectorizer, target_columns)

    app.run()