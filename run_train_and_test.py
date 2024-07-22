from datasets import *
from training_module import train
from testing_module import test, test_recall_at1percent
from models import *
from loss import *
import csv
import os
from config import CONFIG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

def compute_global_error(cloudy_error, night_error, sunny_error):
    total_images = len(test_dataloader_cloudy) + len(test_dataloader_night) + len(test_dataloader_sunny)
    global_error = (len(test_dataloader_cloudy) * cloudy_error + len(test_dataloader_night) * night_error + len(test_dataloader_sunny) * sunny_error) / total_images
    return global_error

def all_errors2global(mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny):
    mae_global = compute_global_error(mae_cloudy, mae_night, mae_sunny)
    var_global = compute_global_error(varianza_cloudy, varianza_night, varianza_sunny)
    desv_global = compute_global_error(desv_cloudy, desv_night, desv_sunny)
    mse_global = compute_global_error(mse_cloudy, mse_night, mse_sunny)
    rmse_global = compute_global_error(rmse_cloudy, rmse_night, rmse_sunny)
    return mae_global, var_global, desv_global, mse_global, rmse_global

def run(model, model_name, train_dataloader, writer):
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    print('Start training of ' + model_name)
    train(model, train_dataloader, model_name, criterion, device, max_epochs=30)
    test_model = torch.load(CONFIG.dataset_folder + 'models/' + model_name + '.pth').to(device)
    recall_cloudy = test_recall_at1percent(test_model, map_data, test_dataloader_cloudy, device)
    recall_night = test_recall_at1percent(test_model, map_data, test_dataloader_night, device)
    recall_sunny = test_recall_at1percent(test_model, map_data, test_dataloader_sunny, device)
    recall = compute_global_error(recall_cloudy, recall_night, recall_sunny)
    writer.writerow([model_name, recall_cloudy, recall_night, recall_sunny, recall])


if __name__ == '__main__':
    if not os.path.exists(CONFIG.dataset_folder + 'models/'):
        os.makedirs(CONFIG.dataset_folder + 'models/')
        print(f"Carpeta '{CONFIG.dataset_folder + 'models/'}' creada.")
    else:
        print(f"Carpeta '{CONFIG.dataset_folder + 'models/'}' ya existe.")

    if not os.path.exists(CONFIG.dataset_folder + 'results/'):
        os.makedirs(CONFIG.dataset_folder + 'results/')
        print(f"Carpeta '{CONFIG.dataset_folder + 'results/'}' creada.")
    else:
        print(f"Carpeta '{CONFIG.dataset_folder + 'results/'}' ya existe.")

    results = CONFIG.dataset_folder + 'results/results_recall.csv'
    datasets = ['s20', 's30', 's40','s60', 's70', 's80']
    with open(results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", 'Recall@1% cloudy', 'Recall@1% night','Recall@1% sunny','Recall@1% Global'])
        for dataset_name in datasets:
            if dataset_name == 's20':
                dataloader = baseline_dataloader_s20
                model = vgg16_500_500_5_s20.to(device)
            elif dataset_name == 's30':
                dataloader = baseline_dataloader_s30
                model = vgg16_500_500_5_s30.to(device)
            elif dataset_name == 's40':
                dataloader = baseline_dataloader_s40
                model = vgg16_500_500_5_s40.to(device)
            elif dataset_name == 's60':
                dataloader = baseline_dataloader_s60
                model = vgg16_500_500_5_s60.to(device)
            elif dataset_name == 's70':
                dataloader = baseline_dataloader_s70
                model = vgg16_500_500_5_s70.to(device)
            elif dataset_name == 's80':
                dataloader = baseline_dataloader_s80
                model = vgg16_500_500_5_s80.to(device)
            else:
                dataloader = baseline_dataloader
                model = vgg16_500_500_5.to(device)

            run(model, dataset_name, dataloader, writer)
