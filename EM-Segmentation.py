import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import cv2
np.random.seed(0)

def EM_Segmentation(data, parameters, epsilon):
    mean_1 = parameters['mean_1']
    mean_2 = parameters['mean_2']
    covariance_1 = parameters['covariance_1']
    covariance_2 = parameters['covariance_2']
    mixing_coefficient_1 = parameters['mixing_coefficient_1']
    mixing_coefficient_2 = parameters['mixing_coefficient_2']
    # initialize loglikelihood
    N_1 = multivariate_normal.pdf(data, mean=mean_1, cov=covariance_1, allow_singular=True)
    N_2 = multivariate_normal.pdf(data, mean=mean_2, cov=covariance_2, allow_singular=True)
    sum_ = mixing_coefficient_1 * N_1 + mixing_coefficient_2 * N_2
    last_loglikelihood = sum(np.log(sum_))

    while(1):
        # Expectation Step
        N_1 = multivariate_normal.pdf(data, mean=mean_1, cov=covariance_1, allow_singular=True)
        N_2 = multivariate_normal.pdf(data, mean=mean_2, cov=covariance_2, allow_singular=True)
        sum_ = mixing_coefficient_1 * N_1 + mixing_coefficient_2 * N_2
        responsibility_1 = mixing_coefficient_1 * N_1 / sum_
        responsibility_2 = mixing_coefficient_2 * N_2 / sum_

        # Maximization Step
        mean_1 = np.sum(responsibility_1.reshape([-1,1]) * data, axis=0) / np.sum(responsibility_1)
        mean_2 = np.sum(responsibility_2.reshape([-1,1]) * data, axis=0) / np.sum(responsibility_2)
        covariance_1 = ((data - mean_1).T).dot((data - mean_1) * responsibility_1.reshape([-1,1])) / np.sum(responsibility_1)
        covariance_2 = ((data - mean_2).T).dot((data - mean_2) * responsibility_2.reshape([-1,1])) / np.sum(responsibility_2)
        mixing_coefficient_1 = np.sum(responsibility_1) / len(data)
        mixing_coefficient_2 = np.sum(responsibility_2) / len(data)

        # Evaluate
        N_1 = multivariate_normal.pdf(data, mean=mean_1, cov=covariance_1, allow_singular=True)
        N_2 = multivariate_normal.pdf(data, mean=mean_2, cov=covariance_2, allow_singular=True)
        sum_ = mixing_coefficient_1 * N_1 + mixing_coefficient_2 * N_2
        loglikelihood = sum(np.log(sum_))
        if (np.abs(loglikelihood - last_loglikelihood) < epsilon):
            break
        else:
            last_loglikelihood = loglikelihood

    parameters['mean_1'] = mean_1
    parameters['mean_2'] = mean_2
    parameters['covariance_1'] = covariance_1
    parameters['covariance_2'] = covariance_2
    parameters['mixing_coefficient_1'] = mixing_coefficient_1
    parameters['mixing_coefficient_2'] = mixing_coefficient_2

    return parameters

def segmentation(data, parameters):
    N_1 = multivariate_normal.pdf(data,
                                  mean=parameters['mean_1'],
                                  cov=parameters['covariance_1'],
                                  allow_singular=True)
    N_2 = multivariate_normal.pdf(data,
                                  mean=parameters['mean_2'],
                                  cov=parameters['covariance_2'],
                                  allow_singular=True)
    responsibility_1 = parameters['mixing_coefficient_1'] * N_1
    responsibility_2 = parameters['mixing_coefficient_2'] * N_2

    return responsibility_1 < responsibility_2

def writeToFile(data, indices, height, width):
    foreground_data = data.copy()
    background_data = data.copy()
    mask_data = data.copy()
    # assume the foreground' size is smaller than background's size
    if (sum(indices) > len(indices)/2.):
        fore_index = [not id for id in indices]
        back_index = indices.copy()
    else:
        back_index = [not id for id in indices]
        fore_index = indices.copy()

    background_data[fore_index] = 0.
    foreground_data[back_index] = 0.
    mask_data[fore_index] = 0.
    mask_data[back_index] = 255.

    print("start write foreground into image......")
    cv2.imwrite('data/foreground.jpg', foreground_data.reshape([height, width, -1]))
    print("succeed in saving foreground image into \"./data/foreground.jpg\"")

    print("start write background into image......")
    cv2.imwrite('data/background.jpg', background_data.reshape([height, width, -1]))
    print("succeed in saving background image into \"./data/background.jpg\"")

    print("start write mask into image......")
    cv2.imwrite('data/mask.jpg', mask_data.reshape([height, width, -1]))
    print("succeed in saving mask image into \"./data/mask.jpg\"")

def initParameters(data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    parameters = {}
    parameters['mean_1'] = centroids[0]
    parameters['mean_2'] = centroids[1]
    parameters['covariance_1'] = np.cov(data[clusters == 0].transpose())
    parameters['covariance_2'] = np.cov(data[clusters == 1].transpose())
    parameters['mixing_coefficient_1'] = 0.4
    parameters['mixing_coefficient_2'] = 0.6
    return parameters

def Process(data):
    height = data.shape[0]
    width = data.shape[1]
    data = data.reshape([-1, 3])
    init_parameters = initParameters(data)
    final_parameters = EM_Segmentation(data, init_parameters, epsilon=1e-4)
    indices = segmentation(data, final_parameters)
    writeToFile(data, indices, height, width)

if __name__ == "__main__":
    image = cv2.imread("data/cow.jpg", cv2.IMREAD_COLOR)
    print("image is loaded.")
    Process(image)


