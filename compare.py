import json

accuracies = {}

try:
    with open('/tmp/nas-metrics.json') as f:
        nas = json.load(f)
        print("Metrics for Neural Architecture Search: ", nas)
    accuracies['nas_acc'] = [float(i['value']) for i in nas if i['name'] == 'accuracy'][0]
except RuntimeError as e:
    print("Error occurred while reading metrics for NAS: ", e)

try:
    with open('/tmp/hyperop-metrics.json') as f:
        hyper = json.load(f)
        print("Metrics for hyper parameter optimization: ", hyper)
    accuracies['hyper_acc'] = [float(i['value']) for i in hyper if i['name'] == 'accuracy'][0]
except RuntimeError as e:
    print("Error occurred while reading metrics for hyperparameter optimization: ", e)

try:
    with open('/tmp/singlemodel-metrics.json') as f:
        fm = json.load(f)
        print("Metrics for model trained with fixed parameters: ", fm)
    accuracies['fm_acc'] = [float(i['value']) for i in fm if i['name'] == 'accuracy'][0]
except RuntimeError as e:
    print("Error occurred while reading metrics for fixed-param model: ", e)

max_acc_name = max(accuracies, key=accuracies.get)
print("Maximum accuracy was {} for {}".format(max(accuracies.values()), max_acc_name))