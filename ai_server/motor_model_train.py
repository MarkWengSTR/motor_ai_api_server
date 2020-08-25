import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pandas as pd
import json
import os

class AI_Motor_Design():

	def __init__(self):
	
		self.rpm=3000
		self.Current=80
		self.Resistance=0.032
		self.model_save_path = "model/"
		self.motor_name = "4kw_motor"
		
		
	def setting_training_data(self):
		Torque_file=os.listdir("training_data/"+self.motor_name+"/Torque/")
		Voltage_file=os.listdir("training_data/"+self.motor_name+"/Voltage/")
		
		for file in Torque_file:
			data_loc="training_data/"+self.motor_name+"/Torque/"+file
			if Torque_file[0]==file:
				Torque_data=pd.read_csv(data_loc)
			else:
				Torque_data=pd.concat([Torque_data,pd.read_csv(data_loc)])


		for file in Voltage_file:
			data_loc="training_data/"+self.motor_name+"/Voltage/"+file
			if Voltage_file[0]==file:
				Voltage_data=pd.read_csv(data_loc)
			else:
				Voltage_data=pd.concat([Voltage_data,pd.read_csv(data_loc)])

		self.Torque_data=Torque_data
		self.Voltage_data=Voltage_data


		
		
class ANN_Training(AI_Motor_Design):

	def __init__(self):
	
		super().__init__()
		super().setting_training_data()
		
		self.data_X=[]
		self.data_Y=[]
		self.model_config={}
			
		self.input_list=['max_power',
						 'eff',
						 'Vrms',
						 'Torque_ripple'
						 ]

		self.output_list=['am',
						  'delta',
						  'R1',
						  'wmt',
						  'wmw'
						 ]
						 
						 
	def ANN_default_process(self):

		self.input_setting()
		self.model_training()
		self.config_save()
		self.output_config()
		
	def do_zscore(self,dataset, save_path="zscore.txt"):
		mean_list = np.mean(dataset, axis=0)
		dataset = dataset - mean_list

		std_list = np.std(dataset, axis=0)
		std_list[std_list == 0] = 1
		dataset = dataset / std_list

		np.savetxt(save_path, np.vstack((mean_list, std_list)))
		return dataset


	def _arrange_data_from_ANSYS_csv(self,pointer):
	
			self.data_parameter={}
			Torque_pointer=self.Torque_data.columns[pointer]
			Voltage_pointer=self.Voltage_data.columns[pointer]
			self.Torque_pointer=Torque_pointer
			
			self.data_parameter={
					'am':float(Torque_pointer.split(' ')[3].split("'")[1][:-3]),
					'delta':float(Torque_pointer.split(' ')[4].split("'")[1][:-3]),
					'R1':float(Torque_pointer.split(' ')[5].split("'")[1][:-2]),
					'wmt':float(Torque_pointer.split(' ')[6].split("'")[1][:-2]),
					'wmw':float(Torque_pointer.split(' ')[7].split("'")[1][:-2]),
					'torque':self.Torque_data[Torque_pointer],
					'VoltageA':self.Voltage_data[Voltage_pointer.split(")")[0][:-1] +"A) " + Voltage_pointer.split(" ")[1] + " " + Torque_pointer.split(' ',2)[2]],
					'VoltageB':self.Voltage_data[Voltage_pointer.split(")")[0][:-1] +"B) " + Voltage_pointer.split(" ")[1] + " " + Torque_pointer.split(' ',2)[2]],
					'VoltageC':self.Voltage_data[Voltage_pointer.split(")")[0][:-1] +"C) " + Voltage_pointer.split(" ")[1] + " " + Torque_pointer.split(' ',2)[2]],
					}
		
	def _setting_calculated_parameter(self):
	
		self.data_parameter['Torque_ripple']=(np.max(self.data_parameter['torque'])-np.min(self.data_parameter['torque']))/np.mean(self.data_parameter['torque'])
		self.data_parameter['max_power']=np.abs(np.mean(self.data_parameter['torque'])*self.rpm*2*np.pi/60)
		self.data_parameter['Copper_loss']=3*(self.Current**2)*self.Resistance
		self.data_parameter['eff']=self.data_parameter['max_power']/(self.data_parameter['max_power']+self.data_parameter['Copper_loss'])
		self.data_parameter['Vrms']=abs(np.sqrt(np.mean((self.data_parameter['VoltageA']-self.data_parameter['VoltageB'])**2)))


	def model_nodes_arrange(self,array,list):

		array.append([self.data_parameter[i] for i in list])
		return array
		
		
		
	def input_setting(self):
	
		for i in range(1,len(self.Torque_data.columns)):
			try:
				self._arrange_data_from_ANSYS_csv(i)
				self._setting_calculated_parameter()
				self.np_X=np.array(self.model_nodes_arrange(self.data_X,self.input_list))
				self.np_Y=np.array(self.model_nodes_arrange(self.data_Y,self.output_list))
			except:
				print("Wrong index:"+self.Torque_pointer)



	def model_training(self):
		self.data_X = np.array(self.data_X).astype(np.float)

		self.data_Y = np.array(self.data_Y).astype(np.float)

		data_X_zscore = self.do_zscore(self.data_X, save_path=self.model_save_path+"/zscore_X.txt")

		data_Y_zscore = self.do_zscore(self.data_Y, save_path=self.model_save_path+"/zscore_Y.txt")
		msk = np.random.rand(len(self.data_X)) < 0.8
		trainX = data_X_zscore[msk]
		testX = data_X_zscore[~msk]

		trainY = data_Y_zscore[msk]
		testY = data_Y_zscore[~msk]


		model_name = self.model_save_path + 'Ansys_test_01.h5'
		loss_name = self.model_save_path + 'Ansys_test_loss_01.csv'
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

		mean_std_X = np.loadtxt(self.model_save_path+"/zscore_X.txt")
		mean_std_Y = np.loadtxt(self.model_save_path+"/zscore_Y.txt")

	def config_save(self):
		self.model_config["training method"]="ANN"
		self.model_config["quantity of data"]=len(self.data_X)
		self.model_config["epochs"]=1000
		self.model_config["motor name"]=self.motor_name

		self.model_config["input_nodes"]={"max_power":{"max":np.max(self.np_X.T[0][:]),"min":np.min(self.np_X.T[0][:])}}
		self.model_config["input_nodes"]["eff"]={"max":np.max(self.np_X.T[1][:]),"min":np.min(self.np_X.T[1][:])}
		self.model_config["input_nodes"]["Vrms"]={"max":np.max(self.np_X.T[2][:]),"min":np.min(self.np_X.T[2][:])}
		self.model_config["input_nodes"]["Torque_ripple"]={"max":np.max(self.np_X.T[3][:]),"min":np.min(self.np_X.T[3][:])}
		self.model_config["input_nodes_quantity"]=len(self.model_config["input_nodes"])


		self.model_config["output_nodes"]={"am":{"max":np.max(self.np_Y.T[0][:]),"min":np.min(self.np_Y.T[0][:])}}
		self.model_config["output_nodes"]["delta"]={"max":np.max(self.np_Y.T[1][:]),"min":np.min(self.np_Y.T[1][:])}
		self.model_config["output_nodes"]["R1"]={"max":np.max(self.np_Y.T[2][:]),"min":np.min(self.np_Y.T[2][:])}
		self.model_config["output_nodes"]["wmt"]={"max":np.max(self.np_Y.T[3][:]),"min":np.min(self.np_Y.T[3][:])}
		self.model_config["output_nodes"]["wmw"]={"max":np.max(self.np_Y.T[4][:]),"min":np.min(self.np_Y.T[4][:])}
		self.model_config["output_nodes_quantity"]=len(self.model_config["output_nodes"])

	def output_config(self):

		ret = json.dumps(self.model_config)
		with open(self.model_save_path+'/model_config.json', 'w') as fp:
			fp.write(ret)
		print(len(self.data_parameter))
		
		
if __name__=="__main__":
	motor=AI_Motor_Design()
	motor.setting_training_data()
	ANN=ANN_Training()
	ANN.ANN_default_process()


