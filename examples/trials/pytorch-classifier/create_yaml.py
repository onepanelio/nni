# script to generate config yaml file dynamically
import yaml
import argparse
import json

def main(args):
    stream = open(args['config_path'], 'r')
    data = yaml.load(stream)
    # update command based on args
    data['trial']['command'] = "python3 main.py --num_classes {} --epochs {} --model_type {}".format(args['num_classes'], args['epochs'], args['model_type'])
    # update config settings
    if 'max_trial_num' in args:
        data['maxTrialNum'] = int(args['max_trial_num'])
    if 'max_exec_duration' in args:
        data['maxExecDuration'] = args['max_exec_duration']
    if 'trial_concurrency' in args:
        data['trialConcurrency'] = int(args['trial_concurrency'])
    if 'use_annotation' in args:
        data['useAnnotation'] = args['use_annotation']
    if 'tuner' in args:
        data['tuner']['builtinTunerName'] = args['tuner']

    with open(args['output_path'], 'w') as yaml_file:
        yaml_file.write(yaml.dump(data, default_flow_style=False))

    # update search space for hyperparam tuning
    # script needs to be updated for each new param
    json_data = {}
    if 'momentum_range' in args:
        mm_list = [int(item) for item in args['momentum_range'].split(',')]
        json_data['momentum'] = {"_type":"uniform","_value":mm_list}
    if 'lr_list' in args:
        lr_list = [float(item) for item in args['lr_list'].split(',')]
        json_data['lr'] = {"_type":"choice","_value":lr_list}
    if 'batch_size_list' in args:
        bs_list = [int(item) for item in args['batch_size_list'].split(',')]
        json_data['batch_size'] = {'_type':'choice', '_value':bs_list}

    with open(args['output_search_space_path'], 'w') as json_file:
        json.dump(json_data, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Example')
    parser.add_argument("--config_path", type=str,
                        default='/mnt/nni/examples/trials/pytorch-classifier/config.yml', help="train data directory")
    parser.add_argument("--output_path", type=str,
                        default='/mnt/nni/examples/trials/pytorch-classifier/config.yml', help="model to train")
    parser.add_argument("--output_search_space_path", type=str,
                        default='/mnt/nni/examples/trials/pytorch-classifier/search_space.json', help="model to train")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes in the dataset")
    parser.add_argument("--config", default="batch_size_list=16,32,64,128\nlr_list=0.001,0.001\nmomentum_range=0,1\nepochs=10")
    args = parser.parse_args()
    print("Arguments: ", args)
    extras = args.config.split("\n")
    extras_processed = [i.split("#")[0].replace(" ","") for i in extras if i]
    config = {i.split('=')[0]:i.split('=')[1] for i in extras_processed}
    print("[Create YAML] Config: ", config)
    config.update(vars(args))
    print("Final Arguments: ", config)
    main(config)