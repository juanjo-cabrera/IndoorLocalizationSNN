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


def test_recall_5(model, map_data, test_dataloader, device):
    model = model.cuda()
    model.eval()
    freiburg_map = FreiburgMap(map_data, model)
    errors = []
    tp = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            if total%5 == 0:
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

def test_recall_at1percent(model, map_data, test_dataloader, device):
    model = model.cuda()
    model.eval()
    freiburg_map = FreiburgMap(map_data, model)

    tp = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:           
            test_img, test_coor = data[0].to(device), data[1].to(device)
            well_retrieved = freiburg_map.evaluate_recall_at1percent(test_img, test_coor, model)
            if well_retrieved == True:
                tp += 1
            total += 1

    recall = (tp/total) * 100
    print('Recall at 1%:', recall)
    return recall