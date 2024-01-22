from datasets import *
from training_module import train
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
    test_model = torch.load(model_name).to(device)
    mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy = test(test_model, map_data, test_dataloader_cloudy, device)
    mae_night, varianza_night, desv_night, mse_night, rmse_night = test(test_model, map_data, test_dataloader_night, device)
    mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny = test(test_model, map_data, test_dataloader_sunny, device)
    mae_global, var_global, desv_global, mse_global, rmse_global = all_errors2global(mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny)

    writer.writerow([model_name, mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mae_night, varianza_night, desv_night, mse_night, rmse_night, mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, mae_global, var_global, desv_global, mse_global, rmse_global])


if __name__ == '__main__':
    results = '/home/arvc/Juanjo/develop/IndoorLocalizationSNN/table16_resultsb.csv'
    models_names = ['vgg16_500_500_5', 'vgg16_4096_4096_1000', 'vgg16_1000_1000_10']
    with open(results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model_name", "MAE (m) Cloudy", 'Var. Cloudy', 'Desv. Cloudy','MSE Cloudy', 'RMSE Cloudy', "MAE  (m) Night", 'Var. Night', 'Desv. Night', 'MSE Night', 'RMSE Night',"MAE (m) Sunny", 'Var. Sunny', 'Desv. Sunny', 'MSE Sunny', 'RMSE Sunny', "MAE (m) global", 'Var. global', 'Desv. global', 'MSE global', 'RMSE global'])
        for model_name in models_names:
            if model_name == 'vgg16_500_500_5':
                model = vgg16_500_500_5
            elif model_name == 'vgg16_4096_4096_1000':
                model = vgg16_4096_4096_1000
            elif model_name == 'vgg16_1000_1000_10':
                model = vgg16_1000_1000_10
            run(model, model_name, baseline_dataloader, writer)


