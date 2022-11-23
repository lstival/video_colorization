
import tqdm
import cv2 as cv
import numpy as np
import torch

class EdgeExctration():
    def __init__(self, batch_size, img_size) -> None:
        self.img_size = img_size
        self.batch_size = batch_size
        # self.edges = torch.Tensor(1, 1, self.img_size, self.img_size)

    def get_edges(self, dataset, bound_dw=50, bound_up=150):
        self.list_edges = []
        for img in (dataset):
            # print(f"img shape: {img.shape}")
            temp_img = np.uint8(img*255)
            # print(f"temp_img shape: {temp_img.shape}")
            temp_img = np.moveaxis(temp_img, 0, 2)
            tensor_edges = torch.Tensor(cv.Canny(temp_img, bound_dw, bound_up))
            # tensor_edges_3d = torch.cat((tensor_edges,tensor_edges,tensor_edges), 0)

            self.list_edges.append(tensor_edges.squeeze())

            # print(f"tensor_edges shape: {tensor_edges.shape}")
            # print(f"self.edges shape: {self.edges.shape}")
            # self.edges = torch.cat((self.edges, tensor_edges_3d.squeeze()), 0)

        # return torch.FloatTensor(self.edges)
        return self.list_edges
    

