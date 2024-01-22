import numpy as np
import torch
from evaluate import *
from datasets import *
import csv
from table15_experiments import *
# from Train_pos61_Vic import SiameseNetwork
from models import SiameseNetwork

def compute_errors(errors):
    errors_cuadrado = np.power(errors, 2)
    mae = np.mean(errors)
    mse = np.mean(errors_cuadrado)
    rmse = np.sqrt(mse)
    varianza = np.mean(np.power(errors - mae, 2))
    desv = np.sqrt(varianza)
    return mae, varianza, desv, mse, rmse
def test(model, map_data, test_dataloader, device):
    model = model.cuda()
    model.eval()
    freiburg_map = FreiburgMap(map_data, model)
    errors = []
    with torch.no_grad():
        for data in test_dataloader:
            test_img, test_coor = data[0].to(device), data[1].to(device)
            error = freiburg_map.evaluate_error_position(test_img, test_coor, model)
            errors.append(error)

    mae, varianza, desv, mse, rmse = compute_errors(errors)
    print('Mean Absolute Error in test images (m):', mae)
    print('Varianza:', varianza)
    print('Desviacion', desv)
    print('Mean Square Error (m2)', mse)
    print('Root Mean Square Error (m)', rmse)
    return mae, varianza, desv, mse, rmse

def test_recall(model, map_data, test_dataloader, device):
    model = model.cuda()
    model.eval()
    freiburg_map = FreiburgMap(map_data, model)
    errors = []
    tp = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            test_img, test_coor = data[0].to(device), data[1].to(device)
            error, well_retrieved = freiburg_map.evaluate_error_position_and_recall(test_img, test_coor, model)
            errors.append(error)
            if well_retrieved == True:
                tp += 1
            total += 1

    recall = (tp/total) * 100
    mae, varianza, desv, mse, rmse = compute_errors(errors)
    print('Mean Absolute Error in test images (m):', mae)
    print('Varianza:', varianza)
    print('Desviacion', desv)
    print('Mean Square Error (m2)', mse)
    print('Root Mean Square Error (m)', rmse)
    return mae, varianza, desv, mse, rmse, recall


# ruta = '/home/arvc/Juanjo/develop/IndoorLocalizationSNN'
# models_names = ['net_pos61_epoch0', 'net_pos61_epoch5', 'net_pos61_epoch10', 'net_pos61_epoch15', 'net_pos61_epoch20', 'net_pos61_epoch25', 'net_pos61_epoch29']
# # training_sequences_names = ['noDA', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6']
# results = ruta + '/net_pos61_resutls.csv'
# # test_model = SiameseNetwork()
# test_model = torch.load("net_pos61_epoch15").cuda()#Carga el modelo
# #print(test_model)
# with open(results, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Dataset", "MAE (m) Cloudy", 'Var. Cloudy', 'Desv. Cloudy','MSE Cloudy', 'RMSE Cloudy', 'Recall Cloudy', "MAE  (m) Night", 'Var. Night', 'Desv. Night', 'MSE Night', 'RMSE Night', 'Recall Night',"MAE (m) Sunny", 'Var. Sunny', 'Desv. Sunny', 'MSE Sunny', 'RMSE Sunny', 'Recall Sunny',"MAE (m) global", 'Var. global', 'Desv. global', 'MSE global', 'RMSE global', 'Recall global'])
#     for model_name in models_names:
#         # test_model = torch.load(model_name)
#         mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, recall_cloudy = test_recall(net, map_data, test_dataloader_cloudy, device)
#         mae_night, varianza_night, desv_night, mse_night, rmse_night, recall_night = test_recall(test_model, map_data, test_dataloader_night, device)
#         mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, recall_sunny = test_recall(test_model, map_data, test_dataloader_sunny, device)
#         mae_global, var_global, desv_global, mse_global, rmse_global, recall_global = all_errors2global(mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, recall_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, recall_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, recall_sunny)
#         writer.writerow([model_name, mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, recall_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, recall_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, recall_sunny, mae_global, var_global, desv_global, mse_global, rmse_global, recall_global])
