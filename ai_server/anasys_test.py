import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pandas as pd
import os
import json


def do_zscore(dataset, save_path="zscore.txt"):
    mean_list = np.mean(dataset, axis=0)
    dataset = dataset - mean_list

    std_list = np.std(dataset, axis=0)
    std_list[std_list == 0] = 1
    dataset = dataset / std_list

    np.savetxt(save_path, np.vstack((mean_list, std_list)))
    return dataset







def ai_train(ctx):
		
	request_json=ctx
	
	if os.getcwd()[-9:]!='ai_server':
		os.chdir('ai_server')
		
		
	model_save_path = "model/"
	
	model_config_json = model_save_path+"model_config.json"
	with open(model_config_json) as f:
	  model_config = json.load(f)

	model_name = model_save_path + 'Ansys_test_01.h5'
	loss_name = model_save_path + 'Ansys_test_loss_01.csv'
	#
	model_save_checkpoint = ModelCheckpoint(model_name,  # 要儲存的路徑
											monitor='val_loss',  # 要監控的目標
											verbose=1,
											save_best_only=True,  # 儲存最好的模型權重
											period=2)  # 多久監控一次
	#
	log_save_checkpoint = CSVLogger(loss_name, separator=',', append=False)
	#
	early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)


	model = Sequential()
	model.add(Dense(1000, input_shape=(model_config["input_nodes_quantity"],)))
	model.add(Dense(1000))
	model.add(Dense(model_config["output_nodes_quantity"]))
	model.compile(loss='mse', optimizer='adam')


	mean_std_X = np.loadtxt(model_save_path+"zscore_X.txt")
	mean_std_Y = np.loadtxt(model_save_path+"zscore_Y.txt")

	predict_X = np.array([
						 float(ctx["request"]["max_power"]),
						 float(ctx["request"]["eff"]),
						 float(ctx["request"]["Vrms"]),
						 float(ctx["request"]["Torque_ripple"])
	]).reshape(1, -1)
	
	predict_X -= mean_std_X[0]
	predict_X /= mean_std_X[1]
	
	

	predict_Y = model.predict(predict_X).reshape(1, -1)
	predict_Y *= mean_std_Y[1]
	predict_Y += mean_std_Y[0]

	ctx["response"]["ai_response"]["am"]=float(predict_Y[0][0])
	ctx["response"]["ai_response"]["delta"]=float(predict_Y[0][1])
	ctx["response"]["ai_response"]["R1"]=float(predict_Y[0][2])
	ctx["response"]["ai_response"]["wmt"]=float(predict_Y[0][3])
	ctx["response"]["ai_response"]["wmw"]=float(predict_Y[0][4])

#-------check-------


	check_input_list=["max_power","eff","Vrms","Torque_ripple"]
	
	for check_point in check_input_list:
		if model_config["input_nodes"][check_point]["max"]<ctx["request"][check_point] or model_config["input_nodes"][check_point]["min"]>ctx["request"][check_point]:
			ctx["response"]["ai_response"]["Warning"] +=str(check_point) + " out of training range(max:"+str(model_config["input_nodes"][check_point]["max"]) + ", min:"+str(model_config["input_nodes"][check_point]["min"])+")\n"
			
	check_output_list=["am","delta","R1","wmw","wmt"]

	for check_point in check_output_list:
		if model_config["output_nodes"][check_point]["max"]<ctx["response"]["ai_response"][check_point] or model_config["output_nodes"][check_point]["min"]>ctx["response"]["ai_response"][check_point]:
			ctx["response"]["ai_response"]["Warning"] +=str(check_point) + " out of training range(max:"+str(model_config["output_nodes"][check_point]["max"]) + ", min:"+str(model_config["output_nodes"][check_point]["min"])+")\n"










	os.chdir('..')

	return(ctx)
"""
	data_X.append([
				  input_parameter['P_o'],
				  input_parameter['Copper_loss'],
				  input_parameter['eff'],
				  input_parameter['Vrms'],
				  input_parameter['Torque_ripple']
				  ])

	data_Y.append([
				  input_parameter['am'],
				  input_parameter['delta'],
				  input_parameter['R1'],
				  input_parameter['wmt'],
				  input_parameter['wmw']
				  ])

"""

if __name__ == '__main__':

	ctx_json='{\
			"stator_OD_limit": 120,\
			"max_power": 5000,\
			"voltage_dc": 48,\
			"max_torque_nm": 27,\
			"max_speed_rpm": 5000,\
			"export_path": null,\
			"pj_key": null,\
			"res_url": null,\
			"eff":0.9,\
			"Vrms":41,\
			"Torque_ripple":0.02\
			}'
	ctx=json.loads(ctx_json)

	ai_train(ctx)