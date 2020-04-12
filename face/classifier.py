import math
import numpy as np
from matplotlib import pyplot as plt

from .image import Image
from .image import ImageStorage
from .settings import *


def distance(x, y):
    return np.sum((x - y) ** 2)


TEST_SIZE = list(range(1, 10))


def show_images(
        method,
        my_image,
        found_image,
        descriptive_my_image,
        descriptive_found_image,
        params,
):
    plt.subplot(221)
    plt.imshow(my_image, cmap='gray')
    plt.title('My image')

    plt.subplot(222)
    plt.imshow(found_image, cmap='gray')
    plt.title('Detected Point')

    if method not in ['brightness_hist', 'gradient']:
        if method == 'scale':
            params = reshapes[str(params)]
        else:
            params = (params, params)
        descriptive_my_image = descriptive_my_image.reshape(params)
        descriptive_found_image = descriptive_found_image.reshape(params)

        plt.subplot(223)
        plt.imshow(descriptive_my_image, cmap='gray')
        plt.title('My image descriptor')

        plt.subplot(224)
        plt.imshow(descriptive_found_image, cmap='gray')
        plt.title('Detected Point descriptor')
    else:
        plt.subplot(223)
        plt.hist(descriptive_my_image)
        plt.title('My image descriptor')

        plt.subplot(224)
        plt.hist(descriptive_found_image)
        plt.title('Detected Point descriptor')

    plt.show()


class Classifier:
    def __init__(self):
        pass

    def worker(self, image, train_data, with_image, method, params):
        detection_results = []
        for i, face in enumerate(train_data):
            dist = distance(face['value'], image['value'])

            detection_results.append(dist)

        detected_point_ind = np.array(detection_results).argmin()
        found_image = train_data[detected_point_ind]['image']
        descriptive_found_image = train_data[detected_point_ind]['value']
        my_image = image['image']
        descriptive_my_image = image['value']

        if with_image:
            show_images(
                method,
                my_image,
                found_image,
                descriptive_my_image,
                descriptive_found_image,
                params,
            )
        return train_data[detected_point_ind]['human']

    def find_nearest_face(self, method, params, test_size, with_images=False):
        prepared_data = ImageStorage.load_data(method, params)
        train_data, test_indexes = ImageStorage.train_test_split(
            test_size,
        )
        count = 0
        predicted = []
        for i, image in enumerate(prepared_data):
            if image['image_number'] in test_indexes:
                answer = self.worker(
                    image, train_data, with_images, method, params,
                )

                predicted.append(int(answer == image['human']))
                count += 1
        accuracy = sum(predicted) / count
        print(round(accuracy, 3), 'param: ', params, 'test_size: ', test_size)
        return accuracy, predicted

    def vote(self):
        votes_prediction = []
        for method in ['brightness_hist', 'dft', 'dct', 'gradient']:
            current_votes_prediction = []
            results = []
            for test_size in TEST_SIZE:
                accuracy = 0
                for _ in range(5):
                    res, predicted = self.find_nearest_face(method, best_params[method], test_size)
                    accuracy += res
                results.append(accuracy / 5)
                current_votes_prediction.append(predicted)
            votes_prediction.append(current_votes_prediction)
            plt.plot(TEST_SIZE, list(reversed(results)), label=method)

        accuracies = []
        voting_accuracies = np.array(votes_prediction)
        for col in range(voting_accuracies.shape[1]):
            ls_len = len(voting_accuracies[:, col][0])
            res = np.zeros(ls_len)
            for ls in voting_accuracies[:, col]:
                res += np.array(ls)
            count = 0
            for i in res:
                if i >= 2:
                    count += 1
            accuracies.append(count / res.shape[0])
        plt.plot(TEST_SIZE, list(reversed(accuracies)), label='voting')

        plt.legend(title='Methods:')
        plt.xlabel("Train size")
        plt.ylabel("Accuracy")
        plt.show()

    def cross_val(self, method, params, cross_validation_iteration=3, with_images=False):
        accuracies = []
        for test_size in TEST_SIZE:
            for param in params:
                cross = 0
                for _ in range(cross_validation_iteration):
                    val, _ = self.find_nearest_face(method, param, test_size)
                    cross += val
                accuracies.append((cross / cross_validation_iteration, param, test_size))

        accuracies = np.array(accuracies)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(accuracies[:, 1]), np.array(accuracies[:, 2]), np.array(accuracies[:, 0]), c='r',
                   marker='o')
        ax.set_xlabel('Params')
        ax.set_ylabel('Test size')
        ax.set_zlabel('Accuracy')
        plt.show()

        print('CROSS VAL BEST ESTIMATOR', max(accuracies, key=lambda i: i[0]))
