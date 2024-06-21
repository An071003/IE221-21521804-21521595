import Library.Library as Lb

class ToxicityPredictorApp:
    """
        Class App desktop
    """
    def __init__(self, model, vectorizer, target_columns):
        """
        Khởi tạo giá trị của Class
        :param model: mô hình phân loại bình luận xúc phạm
        :param vectorizer:
        """
        self.model = model
        self.vectorizer = vectorizer
        self.target_columns = target_columns
        self.root = Lb.tk.Tk()
        self.root.title("Dự đoán bình luận xúc phạm")
        self.root.geometry("720x580")

        self.label = Lb.tk.Label(self.root, text="Nhập bình luận:")
        self.label.pack(padx=10, pady=10)

        self.entry = Lb.tk.Entry(self.root, width=40)
        self.entry.pack(padx=10, pady=10)

        self.button = Lb.tk.Button(self.root, text="Dư đoán", command=self.click)
        self.button.pack(padx=10, pady=10)

        self.button_reset = Lb.tk.Button(self.root, text="Reset", command=self.reset)
        self.button_reset.pack(padx=10, pady=10)

        self.fig, self.ax = Lb.plt.subplots()
        self.canvas = Lb.FigureCanvasTkAgg(self.fig, master=self.root)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(padx=10, pady=10)


    def click(self):
        """
        Hàm lấy câu bình luận mà người dùng nhập và dự đoán
        """
        sentence = self.entry.get()
        if not sentence:
            Lb.messagebox.showwarning("Vui lòng chờ...", "Hãy nhập bình luận của bạn.")
            return

        # Dự đoán
        prediction = self.model.predict(self.vectorizer, sentence)
        prediction = Lb.np.array(prediction).flatten()

        # Trực quan hóa kết quả
        self.visualisation_prediction(self.target_columns, prediction)

    def visualisation_prediction(self, target_columns, prediction):
        """
        Hàm trực quan hóa kết quả dự đoán trên window
        @param target_columns: các nhãn dự đoán
        @param prediction: kết quả dự đoán
        """
        self.ax.clear()
        colors = Lb.cm.viridis(Lb.np.linspace(0, 1, len(target_columns)))
        self.ax.bar(target_columns, prediction, color=colors)
        self.ax.set_title('prediction')
        self.ax.set_ylabel('precent')
        self.ax.set_xlabel('Labels')
        self.ax.set_xticks(target_columns)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()

    def reset(self):
        """
        hàm reset đồ thị và ô nhập
        """
        self.ax.clear()
        self.canvas.draw()
        self.entry.delete(0, Lb.tk.END)

    def run(self):
        """
        Hàm chạy App
        """
        self.root.mainloop()