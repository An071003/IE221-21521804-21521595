'''
    Module dùng để chạy mô hình và lưu trọng số mô hình
'''
# Import thư viện
import Library.Library as Lb
import Data.Data as Dt
import Model.Model as Md

data = Dt.Data('../jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')

data.preprocess()

model = Md.MyModel()

model.compile()

history = model.train_model(data.train, data.val)

path = '../Save_model/model.weights.h5'

model.save_weights(path)
