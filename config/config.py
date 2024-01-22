import yaml

class Config():
    def __init__(self, yaml_file='/home/arvc/Juanjo/develop/IndoorLocalizationSNN/config/exp_config.yaml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            self.map_dir = config.get('map_dir')
            self.baseline_training_dir = config.get('baseline_training_dir')
            self.DA2_training_dir = config.get('DA2_training_dir')
            self.DA2rot_training_dir = config.get('DA2rot_training_dir')
            self.DA3_training_dir = config.get('DA3_training_dir')
            self.DA3rot_training_dir = config.get('DA3rot_training_dir')
            self.global_test_dir = config.get('global_test_dir')
            self.cloudy_test_dir= config.get('cloudy_test_dir')
            self.night_test_dir = config.get('night_test_dir')
            self.sunny_tes_dir = config.get('sunny_tes_dir')

CONFIG = Config()