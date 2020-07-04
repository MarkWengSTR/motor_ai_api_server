import numpy as np
import json



result_file='predict_result.npy'
format_json='format.json'
response_json='RESPONSE.json'


a=np.load(result_file)


with open(format_json) as f:
  json_data = json.load(f)


json_data['corner_point']["efficiency"]=float(a[0][0])
json_data['corner_point']['torque_ripple']=float(a[0][1])
dump_data=json.dumps(json_data)


with open(response_json, 'w') as fp:
    fp.write(dump_data)



