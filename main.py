# Import thư viện
import Data.Data as Dt
import Model.Model as Md

check = Dt.Data('./jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')

check.analysis_sample()

check.visualisation_analysis()

check.correlations_between_labels()

check.get_nonstop_token()

check.most_common_toxic_words()

check.preprocess()

model = Md.MyModel()

history = model.train_model(check.train, check.val)

model.visualisation_train_model(history)

evalution = model.evaluate_model(check.test)

model.visualisation_evalution(evalution)

prediction = model.predict(check.vectorizer, 'You freaking suck! I am going to hit you.')

model.visualisation_prediction(prediction)