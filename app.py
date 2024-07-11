from flask import render_template, Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
from Encode_Decode_utils.utils import decodeImage
from predict import defect_detection

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    if request.method == "GET":
        return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
   if request.method == "POST":
    image_file = request.files['file']
    print(image_file)
    #decodeImage(image, clApp.filename)
    classifier = defect_detection()
    result = classifier.predictImage(image_file)
    return result
   else:
       print('Loading Error')


if __name__ == "__main__":
    #clApp = ClientApp()
    app.run(debug=True)
    #application.run(host='0.0.0.0', port=8080, debug=True)



#entrypoint: gunicorn -b :$PORT main:app.server --timeout 120
