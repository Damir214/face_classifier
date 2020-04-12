from face.classifier import Classifier
from face.settings import *


def main():
    classifier = Classifier()
    # for algo in ['brightness_hist', 'dft', 'dct', 'gradient']:
    #     # classifier.cross_val(algo, [i for i in range(1, 30, 1)], with_images=True)
    #     classifier.find_nearest_face(algo, best_params[algo], 1, with_images=True)
    # classifier.cross_val('scale', [i / 10 for i in range(1, 11)], with_images=True)

    # classifier.find_nearest_face('scale', 0.2, 1, with_images=True)
    classifier.vote()

if __name__ == '__main__':
    main()
