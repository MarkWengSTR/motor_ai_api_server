from flask import Flask, request, jsonify
from flask_cors import CORS
from validate import spec_present, data_type_validate, spec_keys_validate
import requests
from threading import Thread
from result import result_process
import time
# debug
# import ipdb; ipdb.set_trace()

# example spec json
# {
#     "stator_OD_limit": 120,
#     "max_power":       5000,
#     "voltage_dc":      48,
#     "max_torque_nm":   27,
#     "max_speed_rpm":   5000,
#     "export_path":     None, # this just for spm simulation use
#     "pj_key": None, # this just for spm simulation use
#     "res_url": None,
# }

app = Flask(__name__)
CORS(app)  # local development cors

class BackgroundProcess(Thread):
    def __init__(self, ctx):
        Thread.__init__(self)
        self.ctx = ctx

    def run(self):
        ctx = self.ctx
        response = requests.post(ctx["request"]["res_url"], json=ctx["response"], headers={'Content-type': 'application/json', 'Accept': 'text/plain'})

        print(response.status_code)

@app.route('/motor_ai', methods=["POST"])
def motor_ai():
    ctx = {
        "request": request.get_json(),
        "success_response": {"msg": "start run at background"},
        "error": {
            "validate": {"msg": ""}
            },
        "response": {},
    }

    if spec_present(ctx) and \
            data_type_validate(ctx) and \
            spec_keys_validate(ctx) and \
            result_process(ctx):
        # send result to url in spec
        thread_a = BackgroundProcess(ctx)
        thread_a.start()
        # for local test
        # return jsonify(ctx["response"])
    else:
        return jsonify(ctx["error"]["validate"])

    return jsonify(ctx["success_response"])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
