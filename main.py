import subprocess
import pydantic
from flask import Flask, render_template, request, flash, redirect, send_file
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = '4827c890c1b84580a2efd2fb7257aa8d'
UPLOAD_FOLDER = 'span-aste-pytorch-main/data/banki/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt'}
countfiles = len(os.listdir('results'))


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Не могу прочитать файл')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Нет выбранного файла')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global countfiles
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.txt')
            file.save(file_path)
            countfiles += 1
            result = subprocess.run([
                "python", "span-aste-pytorch-main/test.py",
                "--bert_model", "ai-forever/ruBert-base",
                "--path_to_checkpoint", "model.pt",
                "--path_to_dataset", "span-aste-pytorch-main/data/banki/dataset.txt",
                "--type", "inference",
                "--verbose", "False",
                "--span_maximum_length", "5"
            ], capture_output=True)
            print(result)
            model_output = result.stdout.decode("utf-8")
            return render_template('result.html', model_output=model_output)
    return render_template('main.html')


@app.route('/download')
def download():
    return send_file("results.csv", as_attachment=True)


if __name__ == '__main__':
    app.run()