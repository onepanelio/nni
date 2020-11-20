import xml.etree.ElementTree as ET
import os
import shutil
import argparse
import random
import glob

def main(args):

    tree = ET.parse(args.xml_path)
    root = tree.getroot()

    # create directories
    for label in root.iter('label'):
        os.makedirs(os.path.join(args.data_dir, 'train', label.find('name').text))
        os.makedirs(os.path.join(args.data_dir, 'test', label.find('name').text))
    images_len = len(list(root.iter('tag')))
    test_len = (images_len * args.test_split )// 100
    count = 0
    for img in root.iter('image'):
        #move image
        lbl = img.find('tag').attrib['label']
        if lbl:
            if bool(random.getrandbits(1)) and count <= test_len :
                shutil.move(os.path.join(args.image_dir, img.attrib['name']), os.path.join(args.data_dir, 'test', lbl, img.attrib['name']))
                count += 1
            else:
                shutil.move(os.path.join(args.image_dir, img.attrib['name']), os.path.join(args.data_dir, 'train', lbl, img.attrib['name']))


def train_test_split(args):

    for dir in os.listdir('/mnt/data/datasets'):
        os.makedirs(os.path.join(args.data_dir, 'train', dir))
        os.makedirs(os.path.join(args.data_dir, 'test', dir))
        a = glob.glob('/mnt/data/datasets/'+dir+'/*.jpg')
        a.extend(glob.glob('/mnt/data/datasets/'+dir+'/*.png'))
        test_len = (len(a) * int(args.test_split) )// 100
        count = 0
        for file in a:
            print(file)
            img_path = os.path.split(file)[-1]
            if bool(random.getrandbits(1)) and count <= test_len:
                shutil.move(file, os.path.join(args.data_dir, 'test', dir, img_path))
                count += 1
            else:
                shutil.move(file, os.path.join(args.data_dir, 'train', dir, img_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', default='/mnt/data/datasets/annotations/default.xml')
    parser.add_argument('--data_dir', default='/mnt/data/datasets/processed_data')
    parser.add_argument('--image_dir', default='/mnt/data/datasets/images')
    parser.add_argument('--test_split', default=20, type=int)
    parser.add_argument('--skip', default="false")
    args = parser.parse_args()
    if args.skip == "false":
        main(args)
    else:
        train_test_split(args)