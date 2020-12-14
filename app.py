from flask import Flask, render_template, request, redirect
import random
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', people_list = student_list, num_students = len(student_list))

@app.route('/checkin_page', methods=['GET'])
def render_checkin_page():
    return render_template('checkin.html')

@app.route('/checkin', methods=['GET'])
def checkin():
    if request.method == 'POST':
        student_name = request.values['student_name']
        student_list.append(student_name)
        return redirect('/')
    else:
        return "Hey you, dont be bad"

@app.route('/delete', methods=["POST", "GET"])
def delete():
    if request.method == "POST":
        student_list.pop()
        return redirect('/')
    else:
        return "Hey you, don't be tricky!"

# video must be in static folder
UPLOAD_FOLDER = 'E:/CoderSchool_Final_Project/super_resolution_video/flask_app/static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods = ['POST'])
def upload():
    global filename, video, fps, width, height

    file = request.files['file']

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

