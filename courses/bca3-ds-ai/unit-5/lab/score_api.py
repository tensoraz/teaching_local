"""
Unit 5 Lab · A real-time scoring API in about 50 lines (standard library + joblib, no web framework needed).

Run   : python score_api.py            (needs credit_model.joblib in the same folder)
Score : send a POST to http://127.0.0.1:8500/score with one application as JSON, for example with Python:

    import json, urllib.request
    app = {"age": 25, "years_employed": 2, "income_lakh": 3.05, "loan_amount_lakh": 6.65, "emi_to_income": 0.567,
           "credit_score": 664, "past_defaults": 0, "employment": "Self-employed"}
    req = urllib.request.Request("http://127.0.0.1:8500/score", json.dumps(app).encode(), {"Content-Type": "application/json"})
    print(json.load(urllib.request.urlopen(req)))

Stop  : Ctrl + C
"""
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = joblib.load(HERE / "credit_model.joblib")       # load ONCE at start-up, not on every request
MODEL_VERSION = "credit-logit-v1"
THRESHOLD = 0.25
FEATURES = ["age", "years_employed", "income_lakh", "loan_amount_lakh", "emi_to_income", "credit_score", "past_defaults", "employment"]


class ScoreHandler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        start = time.perf_counter()
        if self.path != "/score":
            return self.reply(404, {"error": "use POST /score"})
        try:
            app = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            missing = [f for f in FEATURES if f not in app]
            if missing:                                       # validate input before scoring
                return self.reply(400, {"error": "missing fields", "fields": missing})
            row = pd.DataFrame([{f: app[f] for f in FEATURES}])
            p = float(MODEL.predict_proba(row)[0, 1])
        except (ValueError, TypeError) as e:
            return self.reply(400, {"error": str(e)})
        self.reply(200, {"p_default": round(p, 4), "decision": "approve" if p < THRESHOLD else "refer to officer",
                         "threshold": THRESHOLD, "model_version": MODEL_VERSION,
                         "latency_ms": round((time.perf_counter() - start) * 1000, 2)})

    def log_message(self, fmt, *args):                       # one short log line per request (useful for monitoring)
        print(time.strftime("%H:%M:%S"), self.address_string(), fmt % args)


if __name__ == "__main__":
    print("Scoring API on http://127.0.0.1:8500/score  (Ctrl + C to stop)")
    HTTPServer(("127.0.0.1", 8500), ScoreHandler).serve_forever()
