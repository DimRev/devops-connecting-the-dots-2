from flask import Flask, request, jsonify
import boto3

app = Flask(__name__)

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    text = request.form['text']
    comprehend = boto3.client('comprehend')
    result = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return jsonify(result)

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    file = request.files['image']
    s3 = boto3.client('s3')
    rekognition = boto3.client('rekognition')

    s3.upload_fileobj(file, 'final-bucket-dimrev', file.filename)
    response = rekognition.detect_labels(
        Image={'S3Object': {'Bucket': 'final-bucket-dimrev', 'Name': file.filename}},
        MaxLabels=10
    )
    return jsonify(response)

@app.route('/')
def index():
    return '''
    <h1>Flask App</h1>
    <form action="/analyze-text" method="post">
        <label for="text">Text</label>
        <input type="text" name="text" id="text">
        <button type="submit">Analyze Text</button>
    </form>
    <form action="/analyze-image" method="post" enctype="multipart/form-data">
        <label for="image">Image</label>
        <input type="file" name="image" id="image">
        <button type="submit">Analyze Image</button>
        <p>Note: This will take a few seconds to process.</p>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
