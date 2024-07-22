import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch
from config import CONFIG
from operator import itemgetter

class FreiburgMap():
    def __init__(self, map_dset, model):
        self.map_dset = map_dset
        self.get_whole_map()
        self.compute_whole_vectors(model)
    def get_coordinates(self, imgs_tuple):
        map_coordinates = []
        for img_tuple in imgs_tuple:
            ruta = img_tuple[0]
            x_index = ruta.index('_x')
            y_index = ruta.index('_y')
            a_index = ruta.index('_a')
            x=ruta[x_index+2:y_index]
            y=ruta[y_index+2:a_index]
            coor_list= [x,y]
            coor = torch.from_numpy(np.array(coor_list,dtype=np.float32))
            map_coordinates.append(coor)
        return map_coordinates
    def get_whole_map(self):
        self.map_coordinates = self.get_coordinates(self.map_dset.imgs)

    def get_latent_vector(self, image, model):
        image = image.cuda().unsqueeze(0)
        latent_vector = model(image)
        return latent_vector

    def compute_whole_vectors(self, model):
        map_vectors = []
        with torch.no_grad():
            for img in self.map_dset:
                img = img[0]
                latent_vector = self.get_latent_vector(img, model)
                map_vectors.append(latent_vector)
            self.map_vectors = map_vectors

    def load_whole_vectors(self):
        map_vectors = self.map_vectors
        return map_vectors

    def evaluate_error_position(self, test_img, coor_test, model):
        test_vector = model(test_img)
        map_vectors = self.load_whole_vectors()
        distances = []
        for map_vector in map_vectors:
            euclidean_distance = F.pairwise_distance(test_vector, map_vector, keepdim=True)
            distances.append(euclidean_distance)
        ind_min = distances.index(min(distances))

        coor_map = self.map_coordinates[ind_min]
        error_localizacion = F.pairwise_distance(coor_test, coor_map.cuda())
        return error_localizacion.detach().cpu().numpy()

    def evaluate_error_position_and_recall(self, test_img, coor_test, model):
        test_vector = model(test_img)
        map_vectors = self.load_whole_vectors()
        descriptor_distances = []
        coordinate_distances = []
        well_retrieved = False
        i = 0
        for map_vector in map_vectors:
            map_coordinate = self.map_coordinates[i]
            descriptor_euclidean_distance = F.pairwise_distance(test_vector, map_vector, keepdim=True)
            coordinate_euclidean_distance = F.pairwise_distance(coor_test, map_coordinate.cuda(), keepdim=True)
            descriptor_distances.append(descriptor_euclidean_distance)
            coordinate_distances.append(coordinate_euclidean_distance)
            i =+ 1
        ind_min = descriptor_distances.index(min(descriptor_distances))
        ind_coordinate_min = coordinate_distances.index(min(coordinate_distances))
        if ind_min == ind_coordinate_min:
            well_retrieved = True
        coor_map = self.map_coordinates[ind_min]
        error_localizacion = F.pairwise_distance(coor_test, coor_map.cuda())
        return error_localizacion.detach().cpu().numpy(), well_retrieved

    def evaluate_recall_at1percent(self, test_img, coor_test, model):
        test_vector = model(test_img)
        map_vectors = self.load_whole_vectors()
        map_vectors = list(torch.cat(map_vectors).detach().cpu().numpy())
        map_coordinates = list(torch.vstack(self.map_coordinates).numpy())
        descriptor_distances = []
        coordinate_distances = []
        well_retrieved = False
        i = 0
        # create a KDTRee with the map vectors
        from sklearn.neighbors import KDTree
        descriptors_tree = KDTree(map_vectors)
        coordinates_tree = KDTree(map_coordinates)
        # get the 1% of the map vectors
        n = round(len(map_vectors)*0.01)
        # retrieve the 1% nearest neighbors
        descriptor_distances, descriptor_indices = descriptors_tree.query(test_vector.detach().cpu().numpy(), k=n)
        # retrieve the 1% nearest coordinates
        coor_distances, coor_indices = coordinates_tree.query(coor_test.detach().cpu().numpy(), k=1)
        # check if the nearest neighbors are the same
        if coor_indices in descriptor_indices:
            well_retrieved = True
            
        return well_retrieved
        

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()


map_data = dset.ImageFolder(root=CONFIG.map_dir, transform=transforms.ToTensor())
map_dataloader = DataLoader(map_data,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)