
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder="../frontend")
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = BASE_DIR
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ============================================================
# LOAD REAL ML MODELS
# ============================================================

academic_model = joblib.load(
    os.path.join(MODEL_DIR, "academic_forecast_model.pkl")
)

decline_model = joblib.load(
    os.path.join(MODEL_DIR, "academic_decline_model.pkl")
)

placement_model = joblib.load(
    os.path.join(MODEL_DIR, "placement_model.pkl")
)

print("✅ Academic Forecast model loaded")
print("✅ Academic Decline model loaded")
print("✅ Placement model loaded")


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def frontend_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "application": "PLACEMIND AI",
        "models": {
            "academic_forecast": "loaded",
            "academic_decline": "loaded",
            "placement": "loaded"
        }
    })


# ============================================================
# COMPLETE AI ANALYSIS
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json() or {}

        academic = data.get("academic", {})
        coding = data.get("coding", {})
        technical = data.get("technical", {})
        aptitude = data.get("aptitude", {})

        # ====================================================
        # ACADEMIC INPUT
        # ====================================================

        academic_input = pd.DataFrame([{
            "SEM 1": float(academic["sem1"]),
            "SEM 2": float(academic["sem2"]),
            "SEM 3": float(academic["sem3"]),
            "SEM 4": float(academic["sem4"])
        }])

        sem1 = float(academic["sem1"])
        sem2 = float(academic["sem2"])
        sem3 = float(academic["sem3"])
        sem4 = float(academic["sem4"])

        # ====================================================
        # REAL ACADEMIC FORECAST
        # ====================================================

        academic_forecast = float(
            academic_model.predict(
                academic_input
            )[0]
        )

        # ====================================================
        # REAL ACADEMIC DECLINE MODEL
        # ====================================================

        decline_prediction = int(
            decline_model.predict(
                academic_input
            )[0]
        )

        decline_probability = None

        if hasattr(
            decline_model,
            "predict_proba"
        ):

            probabilities = (
                decline_model.predict_proba(
                    academic_input
                )[0]
            )

            classes = list(
                decline_model.classes_
            )

            if 1 in classes:

                decline_probability = float(
                    probabilities[
                        classes.index(1)
                    ]
                )

        if decline_probability is not None:

            if decline_probability >= 0.70:

                academic_status = (
                    "High Risk of Academic Decline"
                )

            elif decline_probability >= 0.40:

                academic_status = (
                    "Moderate Risk of Academic Decline"
                )

            else:

                academic_status = (
                    "Low Risk of Academic Decline"
                )

        else:

            academic_status = (
                "Academic trajectory analyzed"
            )

        # ====================================================
        # ACADEMIC READINESS
        #
        # Uses actual SEM 1–4 performance.
        # Latest semester gets slightly higher weight.
        # ====================================================

        academic_average = (
            sem1 * 0.15 +
            sem2 * 0.20 +
            sem3 * 0.25 +
            sem4 * 0.40
        )

        academic_score = min(
            100,
            max(
                0,
                academic_average * 10
            )
        )

        # ====================================================
        # CODING
        # ====================================================

        coding_score = min(
            100,
            max(
                0,
                float(
                    coding.get(
                        "codingScore",
                        0
                    )
                )
            )
        )

        # ====================================================
        # TECHNICAL
        # ====================================================

        technical_score = min(
            100,
            max(
                0,
                float(
                    technical.get(
                        "technicalScore",
                        0
                    )
                )
            )
        )

        # ====================================================
        # APTITUDE
        # ====================================================

        aptitude_score = min(
            100,
            max(
                0,
                (
                    float(
                        aptitude.get(
                            "aptitudeScore",
                            0
                        )
                    )
                    +
                    float(
                        aptitude.get(
                            "communicationScore",
                            0
                        )
                    )
                    +
                    float(
                        aptitude.get(
                            "interviewScore",
                            0
                        )
                    )
                ) / 3
            )
        )

        interview_score = float(
            aptitude.get(
                "interviewScore",
                0
            )
        )

        # ====================================================
        # FINAL READINESS
        # ====================================================

        readiness = round(

            academic_score * 0.25 +

            coding_score * 0.25 +

            technical_score * 0.20 +

            aptitude_score * 0.30,

            1
        )

        if readiness >= 80:

            readiness_label = (
                "HIGH PLACEMENT READINESS"
            )

        elif readiness >= 60:

            readiness_label = (
                "MODERATE PLACEMENT READINESS"
            )

        else:

            readiness_label = (
                "NEEDS IMPROVEMENT"
            )

        # ====================================================
        # WEAKEST AREA
        # ====================================================

        scores = {

            "Academic":
                academic_score,

            "Coding":
                coding_score,

            "Technical":
                technical_score,

            "Aptitude":
                aptitude_score
        }

        weakest_area = min(
            scores,
            key=scores.get
        )

        recommendation = (

            f"Your weakest area is "
            f"{weakest_area}. "

            f"Focus your preparation there "
            f"to improve overall placement "
            f"readiness."

        )

        # ====================================================
        # FORECAST TEXT
        # ====================================================

        forecast_text = (

            f"Predicted next-semester GPA: "
            f"{academic_forecast:.2f}. "

            f"{academic_status}."

        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "readinessScore":
                readiness,

            "readinessLabel":
                readiness_label,

            "academicResult":
                round(
                    academic_score,
                    1
                ),

            "codingResult":
                round(
                    coding_score,
                    1
                ),

            "technicalResult":
                round(
                    technical_score,
                    1
                ),

            "aptitudeResult":
                round(
                    aptitude_score,
                    1
                ),

            "interviewResult":
                round(
                    interview_score,
                    1
                ),

            "academicForecast":
                academic_forecast,

            "academicDeclineProbability":
                decline_probability,

            "academicStatus":
                academic_status,

            "forecast":
                forecast_text,

            "recommendation":
                recommendation

        })

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "error":
                "Prediction failed",

            "details":
                str(e)

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
