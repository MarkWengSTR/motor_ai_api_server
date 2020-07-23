from flask import Flask, request, jsonify
from flask_cors import CORS
from validate import spec_present, data_type_validate, spec_keys_validate
import requests
from result import result_process
from ai_server.anasys_test import ai_train
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

@app.route('/motor_ai', methods=["POST"])
def motor_ai():
    ori_request_data = request.get_json()

    ctx = {
        "request": ori_request_data,
        "ai_data": {
            **ori_request_data,
                "max_power": 4000,
                "copper_loss": 200,
                "eff":0.9,
                "Vrms":41,
                "Torque_ripple":0.02
        },

        "error": {
            "validate": {"msg": ""}
            },
        "response":{
            "ai_response":{
                "am":None,
                "delta":None,
                "R1":None,
                "wmt":None,
                "wmw":None,
                "Warning": ""
                },
            "pj_key": ori_request_data["pj_key"],
            "stator_OD": 62.5,
            "motor_length": 150,
            "coil_turn":3,
            "model_picture_path": None,
            "ele_ang_x_axis": [],
            "corner_point": {
                "current": 80,
                "speed": 3000,
                "torque_data": [],
                "avg_torque": None,
                "torque_ripple": None,
                "line_voltage_rms": None,
                "core_loss": None,
                "core_loss_factor": 1,
                "copper_loss": None,
                "efficiency": None,
                "output_power": None,
                "current_density": None,
            },
            "noload": {
                "ph_voltage_data": [],
                "cogging_data": [],
                "ph_voltage_rms": None,
                "cogging": None,
                "speed": 1000,
            },
            "max_speed": {
                "line_voltage_rms": 30.94828758355385,
                "speed": 5000,
            },
            "material_name": {
                "stator": None,
                "rotor": None,
                "magnet": None
            }
        },
    }

    if spec_present(ctx) and \
            data_type_validate(ctx) and \
            spec_keys_validate(ctx):

        ai_train(ctx) and \
            result_process(ctx)

        # send result to url in spec
        response = requests.post(ctx["request"]["res_url"], json=ctx["response"], headers={'Content-type': 'application/json', 'Accept': 'text/plain'})

        print(response.content)

        # for local test
        return jsonify(ctx["response"])
    else:
        return jsonify(ctx["error"]["validate"])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
