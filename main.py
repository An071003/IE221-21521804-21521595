import Library.Library as Lb
import Model.Model as Md
import hashlib

# Tải Model
model = Md.MyModel()
model.build((None, 1800))
model.load_weights('./Save_model/model.weights.h5')

# Cài đặt lớp vector hóa từ vocabulary đã lưu
with open('./Save_model/vectorizer.pkl', 'rb') as f:
    vocabulary = Lb.pkl.load(f)

vectorizer = Lb.Vectorize(vocabulary=vocabulary)

# Khởi tạo web
app = Lb.Flask(__name__)
app.secret_key = '123456789'

def load_data():
    """
    Hàm Load dữ liệu các bình luận và bài viết
    @return: dữ liệu trong file Json
    """
    with open('data.json', 'r', encoding='utf-8') as f:
        data = Lb.json.load(f)
    return data

def save_data(data):
    """
    Hàm lưu lại dữ liệu với khi nhập câu bình luận mới
    @param data: dữ liệu cần lưu
    """
    with open('data.json', 'w', encoding='utf-8') as f:
        Lb.json.dump(data, f, ensure_ascii=False, indent=4)

def load_users():
    """
    Hàm load thông tin người dùng
    @return: dữ liệu người dùng từ file JSON
    """
    with open('users.json', 'r', encoding='utf-8') as f:
        users = Lb.json.load(f)
    return users

def save_users(users):
    """
    Hàm lưu thông tin người dùng
    @param users: dữ liệu người dùng cần lưu
    """
    with open('users.json', 'w', encoding='utf-8') as f:
        Lb.json.dump(users, f, ensure_ascii=False, indent=4)

def hash_password(password):
    """
    Hàm băm mật khẩu
    @param password: mật khẩu gốc
    @return: mật khẩu đã băm
    """
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if Lb.request.method == 'POST':
        email = Lb.request.form['email']
        password = Lb.request.form['password']
        users = load_users()
        for user in users['users']:
            if user['email'] == email and user['password'] == password:
                Lb.session['username'] = user['username']
                return Lb.redirect(Lb.url_for('index'))
        error = 'Invalid email or password'
    return Lb.render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if Lb.request.method == 'POST':
        email = Lb.request.form['email']
        username = Lb.request.form['username']
        password = Lb.request.form['password']
        users = load_users()
        for user in users['users']:
            if user['email'] == email:
                error = 'Email already exists'
                return Lb.render_template('register.html', error=error)
            if user['username'] == username:
                error = 'Username already exists'
                return Lb.render_template('register.html', error=error)
        users['users'].append({'email': email, 'username': username, 'password': password})
        save_users(users)
        return Lb.redirect(Lb.url_for('login'))
    return Lb.render_template('register.html', error=error)

@app.route('/chat', methods=['GET', 'POST'])
def index():
    data = load_data()
    post = data['posts'][0]
    warning = None
    username = Lb.session.get('username')
    if not username:
        return Lb.redirect(Lb.url_for('login'))
    if Lb.request.method == 'POST':
        comment = Lb.request.form['comment']
        if username and comment:
            # Predict using the model
            prediction = model.predict(vectorizer, comment)
            prediction = Lb.np.array(prediction).flatten()
            target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            ban = "Your comment is "
            for i in range(len(prediction)):
                if prediction[i] > 0.5:
                    ban += target[i] + ", "
            if ban != "Your comment is ":
                ban = ban[:-2] + "."
                warning = ban
            else:
                post['comments'].append({'username': username, 'comment': comment})
                save_data(data)
                return Lb.redirect(Lb.url_for('index'))
    return Lb.render_template('index.html', post=post, warning=warning, username=username)

if __name__ == '__main__':
    app.run(debug=True)
