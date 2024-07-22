import yaml
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'exp_config.yaml')):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            self.dataset_folder = config.get('dataset_folder')
            self.map_dir = self.dataset_folder + config.get('map_dir')
            self.baseline_training_dir = self.dataset_folder + config.get('baseline_training_dir')
            self.DA2_training_dir = self.dataset_folder + config.get('DA2_training_dir')
            self.DA2rot_training_dir = self.dataset_folder + config.get('DA2rot_training_dir')
            self.DA3_training_dir = self.dataset_folder + config.get('DA3_training_dir')
            self.DA3rot_training_dir = self.dataset_folder + config.get('DA3rot_training_dir')
            self.global_test_dir = self.dataset_folder + config.get('global_test_dir')
            self.cloudy_test_dir = self.dataset_folder + config.get('cloudy_test_dir')
            self.night_test_dir = self.dataset_folder + config.get('night_test_dir')
            self.sunny_tes_dir = self.dataset_folder + config.get('sunny_tes_dir')

CONFIG = Config()