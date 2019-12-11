from flask import Flask, jsonify, request
from KNN_implementation import predict

HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def flask_app():
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def server_is_up():
        return 'knn server is up'

    @app.route('/predict_knn', methods=['POST'])
    def start():
        print("test*****************************")
        file_name = request.form['name']
        return predict(file_name)
    return app


if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='127.0.0.1')