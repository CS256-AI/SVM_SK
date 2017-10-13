import sys
from model.SVM import SVM
import zener_generator as gen


def train(epsilon, max_updates, class_letter, train_folder):
    d = gen.Data()
    _lambda, xp, xn = d.scale_data(d.get_data("generated_images", class_letter))
    #print xp
    #print xn
    model = SVM(xp, xn, epsilon, max_updates)
    model.train()


"""
if len(sys.argv) < 6:
    print("Insufficient number of arguments.\nPattern : python sk_train.py epsilon max_updates class_letter model_file_name train_folder_name")
    sys.exit()
else:
    epsilon = float(sys.argv[1])
    max_updates = int(sys.argv[2])
    class_letter, model_file, train_folder = sys.argv[3:]
"""

train(0.1, 10000, "O", "generated_images")