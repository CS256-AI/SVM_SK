import sys
from model.SVM import SVM
from zener_generator import Data

if len(sys.argv) < 4:
    print("Insufficient number of arguments.\nPattern : python svm_model_tester.py model_file_name train_folder_name test_folder_name")
    sys.exit()
else:
    model_file_name, train_folder_name, test_folder_name = sys.argv[1:]

model = SVM()
model.load_model(model_file_name)

d = Data()
file_names, x, y = d.get_data(test_folder_name)

tp, tn, fp, fn = 0,0,0,0
for idx,f in enumerate(file_names):
    test_type = f.split("_")[1].split(".")[0]
    positive_sample = test_type == model.class_letter

    # Scaling test sample
    if positive_sample:
        x[idx] = d.scale_data(x[idx], model._lambda, model.mpos)
    else:
        x[idx] = d.scale_data(x[idx], model._lambda, model.mneg)

    prediction = model.test(x[idx])
    if positive_sample and prediction == 1:
            result = "Correct"
            tp += 1
    elif not positive_sample and prediction == 0:
            result = "Correct"
            tn += 1
    elif not positive_sample and prediction == 1:
            result = "False Positive"
            fp += 1
    elif positive_sample and prediction == 0:
            result = "False Negative"
            fn += 1

    print "{} {}".format(idx, result)

fraction_correct = (tp+tn)/(len(x)*1.0)
fraction_fp = (fp)/(max((tp+fp)*1.0,1))
fraction_fn = (fn)/(max((fn+tn)*1.0,1))
print "Fraction Correct: {}\nFraction False Positive: {}\nFraction False Negative: {}".format(fraction_correct, fraction_fp, fraction_fn)
