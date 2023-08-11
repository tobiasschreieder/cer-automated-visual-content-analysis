"""
The methods in flask.py follow, with slight variations, the work of:

Braker, J., Heinemann, L., Schreieder, T.: Aramis at touché 2022: Argument detection in pictures using machine learning.
Working Notes Papers of the CLEF (2022)

Carnot, M.L., Heinemann, L., Braker, J., Schreieder, T., Kiesel, J., Fröbe,
M., Potthast, M., Stein, B.: On Stance Detection in Image Retrieval for Argumentation. In: Proc. of SIGIR. ACM (2023)

Schreieder, T., Braker, J.: Touché 2022 Best of Labs: Neural Image Retrieval for Argumentation. CLEF (2023)

The original source code can be found at: https://github.com/webis-de/SIGIR-23
"""

import datetime
import logging
import os
from pathlib import Path
from typing import List

from flask import Flask, render_template, request, send_file, abort, make_response

from annotation.evaluation import save_eval, get_image_to_eval, has_eval
from preprocessing.data_entry import DataEntry, Topic
from config import Config


log = logging.getLogger('annotation.flask')
app = Flask(__name__, static_url_path='', static_folder='static')

valid_system = False
image_ids: List[str] = []
cfg = Config.get()


def start_server(debug: bool = True, host: str = '0.0.0.0', port: int = 5000):
    log.debug('starting Flask')
    app.secret_key = '977e39540e424831d8731b8bf17f2484'
    app.debug = debug
    global image_ids, valid_system

    image_ids = DataEntry.get_image_ids()
    valid_system = True

    app.run(host=host, port=port, use_reloader=False)


@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    user = request.cookies.get('user_name', '')
    topics = Topic.load_all()
    selected_topic = Topic.get(51)
    image = None

    topic_done = False
    topic_len = None
    images_done = None
    done_percent = None

    if request.method == 'POST':
        if 'selected_topic' in request.form.keys() and 'user_name' in request.form.keys():
            user = request.form['user_name']
            try:
                selected_topic = Topic.get(int(request.form['selected_topic']))
            except ValueError:
                pass

        if len(user) > 0:
            if 'image_id' in request.form.keys() and 'topic_correct' in request.form.keys():
                if request.form['topic_correct'] == 'topic-true':
                    topic_correct = True
                else:
                    topic_correct = False

                save_eval(image_id=request.form['image_id'], user=user.replace(' ', ''), topic_correct=topic_correct,
                          topic=selected_topic.number)

    if len(user) > 0:
        image = get_image_to_eval(selected_topic)
        if image is None:
            topic_done = True
        topic_len = len(selected_topic.get_image_ids())
        images_done = 0
        for image_id in selected_topic.get_image_ids():
            if has_eval(image_id, selected_topic.number):
                images_done += 1

        done_percent = round((images_done/topic_len)*100, 2)

    resp = make_response(render_template('evaluation.html', topics=topics, selected_topic=selected_topic,
                                         user_name=user, topic_done=topic_done, image=image,
                                         images_done=images_done, topic_len=topic_len, done_percent=done_percent))
    expire = datetime.datetime.now() + datetime.timedelta(days=90)
    if len(user) > 0:
        resp.set_cookie('user_name', user, expires=expire)
    return resp


def get_abs_data_path(path):
    if not path.is_absolute():
        path = Path(os.path.abspath(__file__)).parent.parent.joinpath(path)
    return path


@app.route('/data/image/<path:image_id>')
def data_image(image_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    return send_file(get_abs_data_path(entry.webp_path))


@app.route('/data/png/<path:image_id>')
def data_image_png(image_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    if entry.png_path.exists():
        return send_file(get_abs_data_path(entry.png_path))
    return 'The png files are not in the data directory. Use the webp images.<br>' \
           f'<a href="/data/image/{image_id}">Webp Image</a>'


@app.route('/data/screenshot/<path:image_id>/<path:page_id>')
def data_snp_screenshot(image_id, page_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    for page in entry.pages:
        if page.url_hash == page_id:
            if page.snp_screenshot.exists():
                return send_file(get_abs_data_path(page.snp_screenshot))
            return 'The screenshot files are not in the data directory. Use the direct site or dom file.<br>' \
                   f'<a href="{page.url}">Website URL</a><br>' \
                   f'<a href="/data/dom/{image_id}/{page_id}">Website DOM</a>'
    return abort(404)


@app.route('/data/dom/<path:image_id>/<path:page_id>')
def data_snp_dom(image_id, page_id):
    if image_id not in image_ids:
        return abort(404)
    entry = DataEntry.load(image_id)
    for page in entry.pages:
        if page.url_hash == page_id:
            return send_file(get_abs_data_path(page.snp_dom))
    return abort(404)
