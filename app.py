from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import mysql.connector
import os
import uuid
from werkzeug.utils import secure_filename
import numpy as np
import keras
from keras import preprocessing
import cv2

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="utkarsh2305",
        database="signature_verification"
    )
    cursor = db.cursor()
except mysql.connector.Error as err:
    print(f"Database connection failed: {err}")
    exit(1)

# Register the euclidean_distance function
@keras.saving.register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return keras.ops.sqrt(keras.ops.sum(keras.ops.square(x - y), axis=1, keepdims=True))

# Load the trained model
MODEL_PATH = os.path.join('model', 'signature_model_final.keras')
model = keras.models.load_model(MODEL_PATH)

# Preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Compare two signature images
def compare_signatures(image1_path, image2_path, threshold=0.5):
    try:
        img1 = preprocess_image(image1_path)
        img2 = preprocess_image(image2_path)
        similarity_score = model.predict([img1, img2])
        if similarity_score < threshold:
            return "Signatures match!", similarity_score[0][0]
        else:
            return "Signatures do not match.", similarity_score[0][0]
    except Exception as e:
        return f"Error comparing signatures: {str(e)}", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        branch_id = request.form['branch_id']

        cursor.execute("SELECT COUNT(*) FROM credentials")
        count = cursor.fetchone()[0] + 1
        employee_id = f"EMP{count:04d}"

        query = "INSERT INTO credentials (employee_id, name, username, employee_email, password, branch_id) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(query, (employee_id, name, username, email, password, branch_id))
        db.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor.execute("SELECT * FROM credentials WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            session['username'] = user[2]
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Try again.", "error")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/add_client', methods=['GET', 'POST'])
def add_client():
    if request.method == 'POST':
        account_number = request.form['account_number']
        account_holder = request.form['account_holder']
        email = request.form['account_holder_email']
        address = request.form['account_holder_address']
        signature = request.files['signature_image']

        if signature:
            filename = f"{account_number}_{uuid.uuid4().hex}{os.path.splitext(signature.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            signature.save(filepath)

            query = "INSERT INTO customer_details (account_number, account_holder, account_holder_email, account_holder_address, signature_image) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, (account_number, account_holder, email, address, filename))
            db.commit()
            return redirect(url_for('dashboard'))
    return render_template('add_client.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route("/check_signature", methods=["GET", "POST"])
def check_signature():
    if request.method == "POST":
        account_number = request.form["account_number"]
        cursor.execute("SELECT * FROM customer_details WHERE account_number = %s", (account_number,))
        customer = cursor.fetchone()

        if customer:
            return redirect(url_for("images", account_number=customer[0], account_holder=customer[1], account_holder_email=customer[2], account_holder_address=customer[3], signature_image=customer[4]))
        else:
            flash("Account number not found!", "error")
    return render_template("check_signature.html")

@app.route("/images")
def images():
    return render_template("images.html", account_number=request.args.get("account_number"), account_holder=request.args.get("account_holder"), account_holder_email=request.args.get("account_holder_email"), account_holder_address=request.args.get("account_holder_address"), signature_image=request.args.get("signature_image"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/upload_img', methods=['POST'], endpoint='upload_img')
def upload():
    if 'file2' not in request.files:
        return "No file uploaded."

    account_number = request.form['account_number']
    file2 = request.files['file2']

    if file2.filename == '':
        return "No file selected."

    cursor.execute("SELECT signature_image FROM customer_details WHERE account_number = %s", (account_number,))
    result = cursor.fetchone()

    if not result:
        return "Account number not found."

    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(str(result[0])))
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
    file2.save(file2_path)

    result_text, score = compare_signatures(file1_path, file2_path)

    return render_template('result.html', result=result_text, score=score, image1=result[0], image2=file2.filename)

if __name__ == "__main__":
    app.run(debug=True)
