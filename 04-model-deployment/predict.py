import pickle

from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    # return float(preds[0])
    return float(preds[0])


app = Flask('car-price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    car_features = request.get_json()
    pred = predict(car_features)

    result = {
        'car_price': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)