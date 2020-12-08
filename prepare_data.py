import xml.etree.ElementTree as ET
import os
import shutil
import argparse
import random
import glob

def main(args):
    """ CVAT's XML format has xml file that contains annotations and paths to images.
        The script requires data to be in PyTorch's ImageFolder folder where there is one directory
        per class.

        This also splits data into train and test set.
    """
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
                # randomly put image into test or train dir
                shutil.move(os.path.join(args.image_dir, img.attrib['name']), os.path.join(args.data_dir, 'test', lbl, img.attrib['name']))
                count += 1
            else:
                shutil.move(os.path.join(args.image_dir, img.attrib['name']), os.path.join(args.data_dir, 'train', lbl, img.attrib['name']))


def train_test_split(args):
    """
        If Images are already in ImageFolder format, then just split images into train and test.
    """
    for dirn in os.listdir(args.image_dir):
        os.makedirs(os.path.join(args.data_dir, 'train', dirn))
        os.makedirs(os.path.join(args.data_dir, 'test', dirn))
        a = glob.glob(args.image_dir+'/'+dirn+'/*.jpg')
        a.extend(glob.glob(args.image_dir+'/'+dirn+'/*.png'))
        test_len = (len(a) * int(args.test_split) )// 100
        count = 0
        for file in a:
            img_path = os.path.split(file)[-1]
            if bool(random.getrandbits(1)) and count <= test_len:
                shutil.move(file, os.path.join(args.data_dir, 'test', dirn, img_path))
                count += 1
            else:
                shutil.move(file, os.path.join(args.data_dir, 'train', dirn, img_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', default='/mnt/data/datasets/annotations/default.xml')
    parser.add_argument('--data_dir', default='/mnt/data/datasets/processed_data')
    parser.add_argument('--image_dir', default='/mnt/data/datasets/images')
    parser.add_argument('--test_split', default=20, type=int)
    parser.add_argument('--skip', default="false")
    args = parser.parse_args()
    if args.skip == "false":
        print("Processing data...")
        main(args)
    else:
        print("Moving files to appropriate directories...")
        train_test_split(args)
        # clean up, lost+found directory causes PyTorch to think there are three classes
        # so, remove it
        try:
            shutil.rmtree(os.path.join(args.data_dir, 'train', 'lost+found'))
            shutil.rmtree(os.path.join(args.data_dir, 'test', 'lost+found'))
        except:
            pass