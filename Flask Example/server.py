from flask import Flask, jsonify, request
from prediction import predict


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def flask_app():
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def server_is_up():
        return 'server is up'

    @app.route('/predict_partner', methods=['POST'])
    def start():
        print("test*****************************")
        to_predict = request.json

        print(to_predict)
        pred = predict(to_predict)
        return jsonify({"predict happiness with partner":pred})
    return app


if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='127.0.0.1')