import xml.etree.ElementTree as ET
import os
import shutil
import argparse

def main(args):

    tree = ET.parse(args.xml_path)
    root = tree.getroot()

    # create directories
    for label in root.iter('label'):
        os.makedirs(os.path.join(args.data_dir, label.find('name').text))

    for img in root.iter('image'):
        #move image
        lbl = img.find('tag').attrib['label']
        if lbl:
            shutil.move(os.path.join(args.image_dir, img.attrib['name']), os.path.join(args.data_dir, lbl, img.attrib['name']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', default='/mnt/data/datasets/annotations/default.xml')
    parser.add_argument('--data_dir', default='/mnt/data/datasets/processed_data')
    parser.add_argument('--image_dir', default='/mnt/data/datasets/images')
    args = parser.parse_args()
    main(args)
