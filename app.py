from flask import Flask, render_template, request
import pickle
#import ml model here

app = Flask(__name__)

with open('best_neural_network_model.pkl', "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_data = request.form.get("input")

        preprocessed_data = input_data

        prediction = model.predict([preprocessed_data])

        predicted_class = prediction[0]

        return render_template("result.html", prediction=predicted_class)
    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)
