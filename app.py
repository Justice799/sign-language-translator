from flask import Flask, render_template, Response, jsonify, request
import datatesting
from generatesign import yield_sign_frames  # Updated import from generatesign.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(datatesting.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    sentence = datatesting.get_sentence()
    return jsonify({'sentence': sentence})

@app.route('/text_to_sign', methods=['GET', 'POST'])
def text_to_sign():
    if request.method == 'POST':
        text = request.form['text']
        return render_template('text_to_sign.html')  # No video_url needed; handled by JS
    return render_template('text_to_sign.html')

@app.route('/sign_video_stream')
def sign_video_stream():
    text = request.args.get('text', '')
    if not text:
        return "No text provided", 400
    return Response(yield_sign_frames(text), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)