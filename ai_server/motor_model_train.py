import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pandas as pd
import json


def do_zscore(dataset, save_path="zscore.txt"):
    mean_list = np.mean(dataset, axis=0)
    dataset = dataset - mean_list

    std_list = np.std(dataset, axis=0)
    std_list[std_list == 0] = 1
    dataset = dataset / std_list

    np.savetxt(save_path, np.vstack((mean_list, std_list)))
    return dataset

model_save_path = "model/"
motor_name = "4kw_motor"

Torque_file = ["Torque Plot 07101030.csv",
			   "Torque Plot 07221130.csv"
			   ]
Voltage_file = ["Winding Plot 07101030.csv",
				"Winding Plot 07221130.csv"
				]

for i in range(len(Torque_file)):
	data_loc="training_data/"+motor_name+"/Torque/"+Torque_file[i]
	if i==0:
		Torque_data=pd.read_csv(data_loc)
	else:
		Torque_data=pd.concat([Torque_data,pd.read_csv(data_loc)])


for i in range(len(Voltage_file)):
	data_loc="training_data/"+motor_name+"/Voltage/"+Voltage_file[i]
	if i==0:
		Voltage_data=pd.read_csv(data_loc)
	else:
		Voltage_data=pd.concat([Voltage_data,pd.read_csv(data_loc)])



rpm=3000
Current=80
Resistance=0.032

data_X=[]
data_Y=[]
for i in range(1,len(Torque_data.columns)):

	Torque_pointer=Torque_data.columns[i]
	Voltage_pointer=Voltage_data.columns[i]
	try:
		input_parameter={
						'am':float(Torque_pointer.split(' ')[3].split("'")[1][:-3]),
						'delta':float(Torque_pointer.split(' ')[4].split("'")[1][:-3]),
						'R1':float(Torque_pointer.split(' ')[5].split("'")[1][:-2]),
						'wmt':float(Torque_pointer.split(' ')[6].split("'")[1][:-2]),
						'wmw':float(Torque_pointer.split(' ')[7].split("'")[1][:-2]),
						'torque':Torque_data[Torque_pointer],
						'VoltageA':Voltage_data[Voltage_pointer.split(")")[0][:-1] +"A) " + Voltage_pointer.split(" ")[1] + " " + Torque_pointer.split(' ',2)[2]],
						'VoltageB':Voltage_data[Voltage_pointer.split(")")[0][:-1] +"B) " + Voltage_pointer.split(" ")[1] + " " + Torque_pointer.split(' ',2)[2]],
						'VoltageC':Voltage_data[Voltage_pointer.split(")")[0][:-1] +"C) " + Voltage_pointer.split(" ")[1] + " " + Torque_pointer.split(' ',2)[2]],
						}

		input_parameter['max_power']=np.abs(np.mean(input_parameter['torque'])*rpm*2*np.pi/60)
		input_parameter['Copper_loss']=3*(Current**2)*Resistance
		input_parameter['eff']=input_parameter['max_power']/(input_parameter['max_power']+input_parameter['Copper_loss'])
		input_parameter['Vrms']=abs(np.sqrt(np.mean((input_parameter['VoltageA']-input_parameter['VoltageB'])**2)))
		input_parameter['Torque_ripple']=(np.max(input_parameter['torque'])-np.min(input_parameter['torque']))/np.mean(input_parameter['torque'])
		
		data_X.append([
					  input_parameter['max_power'],
					#  input_parameter['Copper_loss'],
					  input_parameter['eff'],
					  input_parameter['Vrms'],
					  input_parameter['Torque_ripple']
					  ])
		np_X=np.array(data_X)
		data_Y.append([
					  input_parameter['am'],
					  input_parameter['delta'],
					  input_parameter['R1'],
					  input_parameter['wmt'],
					  input_parameter['wmw']
					  ])
		np_Y=np.array(data_Y)
	except:
		print("Wrong index:"+Torque_pointer)

model_config={}

model_config["input_nodes"]={"max_power":{"max":np.max(np_X.T[0][:]),"min":np.min(np_X.T[0][:])}}
model_config["input_nodes"]["eff"]={"max":np.max(np_X.T[1][:]),"min":np.min(np_X.T[1][:])}
model_config["input_nodes"]["Vrms"]={"max":np.max(np_X.T[2][:]),"min":np.min(np_X.T[2][:])}
model_config["input_nodes"]["Torque_ripple"]={"max":np.max(np_X.T[3][:]),"min":np.min(np_X.T[3][:])}
model_config["input_nodes_quantity"]=len(model_config["input_nodes"])


model_config["output_nodes"]={"am":{"max":np.max(np_Y.T[0][:]),"min":np.min(np_Y.T[0][:])}}
model_config["output_nodes"]["delta"]={"max":np.max(np_Y.T[1][:]),"min":np.min(np_Y.T[1][:])}
model_config["output_nodes"]["R1"]={"max":np.max(np_Y.T[2][:]),"min":np.min(np_Y.T[2][:])}
model_config["output_nodes"]["wmt"]={"max":np.max(np_Y.T[3][:]),"min":np.min(np_Y.T[3][:])}
model_config["output_nodes"]["wmw"]={"max":np.max(np_Y.T[4][:]),"min":np.min(np_Y.T[4][:])}
model_config["output_nodes_quantity"]=len(model_config["output_nodes"])



data_X = np.array(data_X).astype(np.float)

data_Y = np.array(data_Y).astype(np.float)

data_X_zscore = do_zscore(data_X, save_path=model_save_path+"/zscore_X.txt")

data_Y_zscore = do_zscore(data_Y, save_path=model_save_path+"/zscore_Y.txt")


msk = np.random.rand(len(data_X)) < 0.8
trainX = data_X_zscore[msk]
testX = data_X_zscore[~msk]

trainY = data_Y_zscore[msk]
testY = data_Y_zscore[~msk]


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
model.add(Dense(1000, input_shape=(trainX.shape[1],)))
model.add(Dense(1000))
model.add(Dense(trainY.shape[1]))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, verbose=1, validation_data=[testX, testY],
          callbacks=[early_stopping, model_save_checkpoint, log_save_checkpoint])

mean_std_X = np.loadtxt(model_save_path+"/zscore_X.txt")
mean_std_Y = np.loadtxt(model_save_path+"/zscore_Y.txt")


model_config["training method"]="ANN"
model_config["quantity of data"]=len(data_X)
model_config["epochs"]=1000
model_config["motor name"]=motor_name



ret = json.dumps(model_config)
with open(model_save_path+'/model_config.json', 'w') as fp:
	fp.write(ret)







