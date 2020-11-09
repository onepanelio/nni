import xml.etree.ElementTree as ET
import os
import shutil
import argparse
import random

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', default='/mnt/data/datasets/annotations/default.xml')
    parser.add_argument('--data_dir', default='/mnt/data/datasets/processed_data')
    parser.add_argument('--image_dir', default='/mnt/data/datasets/images')
    parser.add_argument('--test_split', default=20, type=int)
    args = parser.parse_args()
    main(args)
