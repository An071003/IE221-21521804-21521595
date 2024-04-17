import Data.Data as dt

check = dt.Data('./jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')

check.Analysis_sample()

check.Visualisation_Analysis()

check.correlations_between_labels()

check.get_nonstop_token()

check.Most_common_toxic_words()

check.preprocess()
