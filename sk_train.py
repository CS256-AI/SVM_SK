import sys
from model.SVM import SVM
import zener_generator as gen


def train(epsilon, max_updates, class_letter, train_folder, model_file_name):
    d = gen.Data()
    _lambda,x_pos, x_neg, m_p, m_n = d.find_scale(d.get_data(train_folder, class_letter))
    xp = d.scale_data(x_pos, _lambda, m_p)
    xn = d.scale_data(x_neg, _lambda, m_n)
    model = SVM(xp, xn, epsilon, max_updates)
    model.train()
    model.save_model(model_file_name)


"""
if len(sys.argv) < 6:
    print("Insufficient number of arguments.\nPattern : python sk_train.py epsilon max_updates class_letter model_file_name train_folder_name")
    sys.exit()
else:
    epsilon = float(sys.argv[1])
    max_updates = int(sys.argv[2])
    class_letter, model_file, train_folder = sys.argv[3:]
"""

train(0.1, 10 0000, "O", "generated_images", "D:\SJSU\Fall17\CS256\SVM_SK\Model_O.txt")