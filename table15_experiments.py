from datasets import *
from training_module import *
from testing_module import test
from models import *
from loss import *
import csv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

def compute_global_error(cloudy_error, night_error, sunny_error):
    total_images = len(test_dataloader_cloudy) + len(test_dataloader_night) + len(test_dataloader_sunny)
    global_error = (len(test_dataloader_cloudy) * cloudy_error + len(test_dataloader_night) * night_error + len(test_dataloader_sunny) * sunny_error) / total_images
    return global_error

def all_errors2global(mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, recall_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, recall_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, recall_sunny):
    mae_global = compute_global_error(mae_cloudy, mae_night, mae_sunny)
    var_global = compute_global_error(varianza_cloudy, varianza_night, varianza_sunny)
    desv_global = compute_global_error(desv_cloudy, desv_night, desv_sunny)
    mse_global = compute_global_error(mse_cloudy, mse_night, mse_sunny)
    rmse_global = compute_global_error(rmse_cloudy, rmse_night, rmse_sunny)
    recall_global = compute_global_error(recall_cloudy, recall_night, recall_sunny)
    return mae_global, var_global, desv_global, mse_global, rmse_global, recall_global

def run(model, model_name, train_dataloader, writer):
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    # if model_name == 'DAs60':
    print('Start training of ' + model_name)
    trainV2(model, train_dataloader, model_name, criterion, device, max_epochs=21)

    epochs = [1, 3, 10, 20]
    print('Start testing of ' + model_name)
    for epoch in epochs:
        model_name_epoch = model_name + '_epoch' + str(epoch)
        test_model = torch.load(model_name_epoch).to(device)
        mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy = test(test_model, map_data, test_dataloader_cloudy, device)
        mae_night, varianza_night, desv_night, mse_night, rmse_night = test(test_model, map_data, test_dataloader_night, device)
        mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny = test(test_model, map_data, test_dataloader_sunny, device)
        mae_global, var_global, desv_global, mse_global, rmse_global = all_errors2global(mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny)

        writer.writerow([model_name_epoch, mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, mae_global, var_global, desv_global, mse_global, rmse_global])


if __name__ == '__main__':
    results = '/home/arvc/Juanjo/develop/IndoorLocalizationSNN/table15_resultsDA3rot.csv'
    datasets = ['DA3rot_s50', 'DA3rot_s60']
    with open(results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "MAE (m) Cloudy", 'Var. Cloudy', 'Desv. Cloudy','MSE Cloudy', 'RMSE Cloudy', "MAE  (m) Night", 'Var. Night', 'Desv. Night', 'MSE Night', 'RMSE Night',"MAE (m) Sunny", 'Var. Sunny', 'Desv. Sunny', 'MSE Sunny', 'RMSE Sunny', "MAE (m) global", 'Var. global', 'Desv. global', 'MSE global', 'RMSE global'])
        for dataset_name in datasets:
            if dataset_name == 'DA3rot_s50':
                dataloader = DA3rot_dataloader_s50
                model = vgg16_500_500_5.to(device)         
            elif dataset_name == 'DA3rot_s60':
                dataloader = DA3rot_dataloader_s60
                model = vgg16_500_500_5_s60.to(device)     

            run(model, dataset_name, dataloader, writer)


