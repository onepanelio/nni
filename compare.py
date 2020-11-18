import json

accuracies = {}
with open('/tmp/nas-metrics.json') as f:
    nas = json.load(f)

with open('/tmp/hyperop-metrics.json') as f:
    hyper = json.load(f)

with open('/tmp/singlemodel-metrics.json') as f:
    fm = json.load(f)

accuracies['nas_acc'] = [float(i['value']) for i in nas if i['name'] == 'accuracy'][0]
accuracies['hyper_acc'] = [float(i['value']) for i in hyper if i['name'] == 'accuracy'][0]
accuracies['fm_acc'] = [float(i['value']) for i in fm if i['name'] == 'accuracy'][0]

max_acc_name = max(accuracies, key=accuracies.get)
print("Maximum accuracy was {} for {}".format(max(accuracies.values()), max_acc_name))