import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pandas as pd

# import json


def do_zscore(dataset, save_path="ai_server/zscore.txt"):
    mean_list = np.mean(dataset, axis=0)
    dataset = dataset - mean_list

    std_list = np.std(dataset, axis=0)
    std_list[std_list == 0] = 1
    dataset = dataset / std_list

    np.savetxt(save_path, np.vstack((mean_list, std_list)))
    return dataset

# POST_json = "POST.json"


# with open(POST_json) as f:
#   json_data = json.load(f)

# f = open(train_file, "r")
# row = f.readlines()
# f.close()

def ai_train(request_json):
    model_save_path = "ai_server/model/"
    train_file = "ai_server/Torque Plot eff2.csv"

    f = open(train_file, "r")
    row = f.readlines()
    f.close()

    row = row[:3]
    data_Y = []

    for i in range(len(row)):
            row[i] = row[i].split(",")[1:]
            row[i][-1] = row[i][-1][:-2]

            if i == 2:
                    for j in range(len(row[i])):
                            temp = row[i][j].split("'")
                            temp2 = [temp[1][:-2], temp[3][:-2], temp[5]]
                            data_Y.append(temp2)



    data_X = np.array(data_Y).astype(np.float)

    data_Y = np.array(row[:2]).T.astype(np.float)

    data_X_zscore = do_zscore(data_X, save_path="ai_server/zscore_X.txt")

    data_Y_zscore = do_zscore(data_Y, save_path="ai_server/zscore_Y.txt")
    print(data_X)

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

    mean_std_X = np.loadtxt("ai_server/zscore_X.txt")
    mean_std_Y = np.loadtxt("ai_server/zscore_Y.txt")

    predict_X = np.array([
                         float(request_json["stator_OD"]),
                         float(request_json["motor_length"]),
                         float(request_json["coil_turn"])
    ]).reshape(1, -1)

    predict_X -= mean_std_X[0]
    predict_X /= mean_std_X[1]

    print(predict_X)

    predict_Y = model.predict(predict_X).reshape(1, -1)
    predict_Y *= mean_std_Y[1]
    predict_Y += mean_std_Y[0]

    print(predict_Y)

    predict_result_file="predict_result"
    np.save(predict_result_file, predict_Y)



