import Library.Library_struct as lb


class Data:
    def __init__(self, path):
        self.df = lb.pd.DataFrame(lb.pd.read_csv(path))
        self.target_columns = list(lb.np.array(self.df.columns)[2:])
        self.analysis_columns = []
        self.nlp = lb.spacy.load('en_core_web_lg')
        self.train = []
        self.test = []
        self.val = []

    def Analysis_sample(self):
        # Thêm một số nhãn cho mục đích phân tích
        self.df['non-toxic'] = 1 - self.df[self.target_columns].max(axis=1)
        self.df['toxicity_type_defined'] = self.df[['insult', 'obscene', 'identity_hate', 'threat']].max(axis=1)
        self.df['toxic_undefined'] = 0
        self.df.loc[(self.df['toxicity_type_defined'] == 0) & (self.df['toxic'] == 1), 'toxic_undefined'] = 1
        self.df['soft_toxic'] = 0
        self.df.loc[(self.df['toxicity_type_defined'] == 1) & (self.df['toxic'] == 0), 'soft_toxic'] = 1
        self.analysis_columns = self.target_columns + ['non-toxic', 'toxic_undefined', 'soft_toxic']

    def Visualisation_analysis(self):
        # Trực quan hóa dữ liệu đã được phân tích
        label_counts = self.df[self.analysis_columns].sum()

        lb.plt.figure(figsize=(20, 10))
        ax = lb.sns.barplot(x=label_counts.index, y=label_counts.values,
                            palette='Set3')  # Set3 is an example of a color palette
        ax.set_yscale('log')
        ax.tick_params(labelsize=15)
        lb.plt.xlabel('Label', fontsize=15)
        lb.plt.ylabel('Count', fontsize=15)
        lb.plt.title('Count of Each Label', fontsize=20)
        lb.plt.show()

    def correlations_between_labels(self):
        # Trực quan hóa các liên kết giữa các labels
        heatmap_data = self.df[self.target_columns]
        lb.plt.figure(figsize=(10, 10))
        ax = lb.sns.heatmap(heatmap_data.corr(), cmap='coolwarm', annot=True)
        ax.tick_params(labelsize=10)
        lb.plt.show()

    def get_nonstop_token(self):
        # Lấy các token không có stop word bên trong
        nonstop_tokens = []
        for doc in self.nlp.pipe(self.df['comment_text'].astype('unicode').values, batch_size=50):
            if doc.has_annotation("DEP"):
                nonstop_tokens.append([t.lower_ for t in doc if t.is_alpha and not t.is_stop])
            else:
                nonstop_tokens.append(None)
        self.df['nonstop_tokens'] = nonstop_tokens

    def Most_common_toxic_words(self):
        # Trực quan hóa những từ thường gặp trong các bình luận xúc phạm
        for label in self.target_columns:
            word_list = list(self.df.loc[self.df[label] == 1, 'nonstop_tokens'].explode())
            most_common = lb.Counter(word_list).most_common(20)
            words = [w for w, _ in most_common]
            counts = [c for _, c in most_common]
            lb.plt.figure(figsize=(20, 10))
            ax = lb.sns.barplot(x=words, y=counts, palette='Set3')
            ax.set_title(f'Label = {label}', fontsize=15)
            ax.tick_params(labelsize=15)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            lb.plt.show()

    def preprocess(self):
        # Vector hóa dữ liệu văn bản
        vt = lb.Vectorize()
        vt.Vectorizer_adapt(self.df['comment_text'])
        vectorized_text = vt.Vectorizer(self.df['comment_text'])

        # Tạo dữ liệu từ vectorized_text và các dữ liệu của target_columns
        dataset = lb.tf.data.Dataset.from_tensor_slices((vectorized_text, self.df[self.target_columns]))
        dataset = dataset.cache()  # Lưu trữ dữ liệu trên cache
        dataset = dataset.shuffle(160000)  # Xác trộn dữ liệu với khung là 160000
        dataset = dataset.batch(16)  # Tạo các batch cho dữ liệu
        dataset = dataset.prefetch(8)  # Cho phép dữ liệu được xử lý trước trong khi mô hình đang đào tạo.
        self.train = dataset.take(int(len(dataset) * .7))
        self.val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
        self.test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))
