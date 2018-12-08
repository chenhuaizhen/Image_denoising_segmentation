import numpy as np
from scipy.stats import norm
import cv2
np.random.seed(0)

def Gibbs_sampling_denoise(data, width, height, iteration=2, J=1, std=1):
    '''
    :param data: matrix includes [1 or -1]
    :param width: image's width
    :param height: image's height
    :param iteration: number of sampling times
    :param J: coupling strength
    :param std: std of noise norm
    :return: updated data
    '''
    def matrixUpdate(mask):
        # left neighbors
        left_matrix = data.copy()
        left_matrix[:,1:] = left_matrix[:,:-1]
        left_matrix[:,0] = data[:,-1]
        # right neighbors
        right_matrix = data.copy()
        right_matrix[:,:-1] = right_matrix[:,1:]
        right_matrix[:, -1] = data[:, 0]
        # upper neighbors
        up_matrix = data.copy()
        up_matrix[1:,:] = up_matrix[:-1,:]
        up_matrix[0,:] = data[-1,:]
        # bottom neighbors
        bottom_matrix = data.copy()
        bottom_matrix[:-1,:] = bottom_matrix[1:,:]
        bottom_matrix[-1,:] = data[0,:]
        neighbours_sum = left_matrix + right_matrix + up_matrix + bottom_matrix
        positive = np.exp(J * neighbours_sum) * norm.pdf(data, loc=1, scale=std)
        negative = np.exp(-J * neighbours_sum) * norm.pdf(data, loc=-1, scale=std)
        p = positive / (positive + negative)
        prob = np.random.uniform(0, 1, size=[height, width])

        data[(prob <= p) * mask] = 1
        data[(prob > p) * mask] = -1

    for i in range(iteration):
        # even's start state
        mask = np.full(width * height, False)
        mask[::2] = True
        matrixUpdate(mask.reshape([height, width]))
        # odd's start state
        mask = np.full(width * height, False)
        mask[1::2] = True
        matrixUpdate(mask.reshape([height, width]))

    return data

def Process(data, iteration, J, std):
    '''
    :param data: image matrix includes [0 or 255]
    :param iteration: number of sampling
    :param J: coupling strength
    :param std: std of noise norm
    :return: save the denoised image
    '''
    width = data.shape[1]
    height = data.shape[0]
    data[data == 255] = 1
    data[data == 0] = -1
    denoised_data = Gibbs_sampling_denoise(data, width, height, iteration=iteration, J=J, std=std)
    print("denoising is done.")
    denoised_data[denoised_data == 1] = 255
    denoised_data[denoised_data == -1] = 0
    print("start write into image......")
    cv2.imwrite('data/denoised-GS.png', denoised_data.astype(np.uint8))
    print("succeed in saving denoised image into \"./data/denoised-GS.png\"")

if __name__ == "__main__":
    image = cv2.imread("data/noise.png", cv2.IMREAD_GRAYSCALE)
    print("noise image is loaded.")
    Process(image.astype(np.float16), iteration=10, J=1, std=3)
