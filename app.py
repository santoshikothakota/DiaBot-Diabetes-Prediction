from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("diabetes_model.pkl")


# ---------------------------
# Home Route
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# Health Value Prediction
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form.get("pregnancies", 0) or 0),
            float(request.form.get("glucose", 0) or 0),
            float(request.form.get("bp", 0) or 0),
            float(request.form.get("skin", 0) or 0),
            float(request.form.get("insulin", 0) or 0),
            float(request.form.get("bmi", 0) or 0),
            float(request.form.get("dpf", 0) or 0),
            float(request.form.get("age", 0) or 0)
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "⚠️ High Risk of Diabetes"
            diet = "Reduce sugar intake and follow a balanced low-carb diet."
            medicine = "Consult a doctor for medical guidance."
        else:
            result = "✅ Low Risk of Diabetes"
            diet = "Maintain a healthy lifestyle."
            medicine = "No immediate medication required."

        return render_template(
            "index.html",
            prediction=result,
            diet=diet,
            medicine=medicine
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------
# Symptom Checker Page
# ---------------------------
@app.route("/symptom-check")
def symptom_check():
    return render_template("symptom_check.html")


# ---------------------------
# Symptom Prediction Logic
# ---------------------------
@app.route("/symptom_predict", methods=["POST"])
def symptom_predict():

    yes_count = 0

    for key, value in request.form.items():
        if value == "yes":
            yes_count += 1

    if yes_count >= 6:
        prediction = "⚠️ High possibility of Diabetes (based on symptoms)"
        diet = "Follow low-sugar diet and consult a doctor."
        medicine = "Seek professional medical advice."
    else:
        prediction = "✅ Low possibility of Diabetes (based on symptoms)"
        diet = "Maintain healthy eating habits."
        medicine = "No immediate medication needed."

    return render_template(
        "symptom_check.html",
        prediction=prediction,
        diet=diet,
        medicine=medicine
    )


# ---------------------------
# Diet Chat Page
# ---------------------------
@app.route("/diet-chat")
def diet_chat():
    return render_template("chat.html")


# ---------------------------
# Chat API (Simple Version)
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()

    if "diet" in user_message:
        reply = "🥗 A balanced diet with low sugar and high fiber is recommended."
    elif "exercise" in user_message:
        reply = "🏃 Regular physical activity helps manage blood sugar levels."
    else:
        reply = "🤖 Please consult a healthcare professional for personalized advice."

    return jsonify({"reply": reply})


# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
