# Author: Kiron Lebeck <kklebeck@cs.washington.edu>
#  A frontend for inference. To run, do: `bazel run :frontend`
from core import inference
from core import utils
import flask
from flask import request
import werkzeug.utils as wutils
import os
from PIL import Image
import tempfile
import zipfile

app = flask.Flask(__name__)


def _count_images(dir_path):
    return len(utils.get_image_paths(dir_path))


@app.route('/')
def home():
    return flask.render_template('home.html')


@app.route('/upload', methods=["POST"])
def upload_files():
    ret_file = request.files['data_file']
    if not ret_file:
        return "No file"
    filename = wutils.secure_filename(ret_file.filename)
    f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ret_file.save(f)
    return flask.redirect(flask.url_for('uploaded_file', filename=filename))


@app.route('/process/<filename>')
def uploaded_file(filename):
    zip_ref = zipfile.ZipFile(
        os.path.join(app.config['UPLOAD_FOLDER'], filename))
    zip_ref.extractall(app.config['UPLOAD_FOLDER'])
    num_imgs = _count_images(app.config['UPLOAD_FOLDER'])
    score_tups = inference.score_image_directory(app.config['UPLOAD_FOLDER'],
                                                 num_imgs)
    results = []
    scores = []
    # Generate and render 128x128 thumbnails.
    for image_path, score in score_tups:
        basename = "temp-" + os.path.basename(image_path)
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], basename)
        with open(thumbnail_path, "wb") as thumbnail_file:
            thumbnail = Image.open(image_path)
            thumbnail = thumbnail.resize((128, 128), Image.ANTIALIAS)
            thumbnail.save(thumbnail_file)
        results.append(basename)
        scores.append(score)
    return flask.render_template(
        'show_images.html', results=results, scores=scores)


@app.route('/uploads/<filename>')
def send_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        app.config['UPLOAD_FOLDER'] = tmp_dir
        app.run(debug=True)
