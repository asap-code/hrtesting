from flask import Flask, request, jsonify
from gpt_model import GPTModel

app = Flask(__name__)
model = GPTModel()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        input_data = data['Content mail']
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction})
    except KeyError:
        return jsonify({'error': 'Invalid input format'}), 400

if __name__ == '__main__':
    # Load and train the model (can be done elsewhere if training is not required on every start)
    training_txt,training_lbs, validation_txt, validation_lbs = model.load_and_process_data("Example mails with category and team.xlsx", random_seed=42)
    model.train(training_txt, training_lbs)
    app.run(host='0.0.0.0')