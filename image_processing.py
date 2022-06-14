import numpy as np
import cv2
import os
from xml.etree import ElementTree as et

class ImageProcessing:

    def getFilePath(self, folder):
        """
        Get filepath from folder
        :param folder: The folder with images and annotation files
        :return: lists of images and annotation paths
        """
        file_path = []
        directory = os.listdir(folder)
        for filename in directory:
            path = folder + "/" + filename  # "dataset/Optisk/190616_112733.jpg"
            file_path.append(path)
        image_path = file_path[0::2]
        annotation_path = file_path[1::2]
        return image_path, annotation_path


    def readImagesInFolder(self, img_path):
        """
        Read the images in folder. Convert to grayscale
        :param img_path: The path to the image
        :return images: Grayscale images
        """
        images = []
        for path in img_path:
            img = cv2.imread(path)  # Read image and convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images.append(gray)
        return images


    def getXYcoorinates(self, x_, tag_name):
        """
        Get xy-coordinates of the ground truth bounding box
        :param x_: the object of annotation file
        :param tag_name: label of the object
        :return: xy-coordinates of ground truth bounding box
        """
        xmin = x_.find('bndbox/' + tag_name[0])
        ymin = x_.find('bndbox/' + tag_name[1])
        xmax = x_.find('bndbox/' + tag_name[2])
        ymax = x_.find('bndbox/' + tag_name[3])

        x = int(xmin.text)
        y = int(ymin.text)
        x_end = int(xmax.text)
        y_end = int(ymax.text)
        return x, y, x_end, y_end


    def saveImageToFolder(self, img_to_save, img_name, folder):
        """
        Save image to folder
        :param img_to_save: the image to save
        :param img_name: the name of the stored image
        :param folder: the folder to save image
        """
        cv2.imwrite(os.path.join(folder, img_name), img_to_save)
        cv2.waitKey(0)


    def cropTrainImages(self, images, label_path):
        """
        Crop images based on ground truth bounding boxes.
        :param images: image to crop
        :param label_path: the path to annotation file
        :return: image list of positive and negative targets
        """
        images_positive = []
        images_negative = []

        tag_name = ['xmin', 'ymin', 'xmax', 'ymax']

        class_name = ['motorboat', 'sailboat', 'barge']

        for i in range(len(images)):
            tree = et.parse(label_path[i])
            myroot = tree.getroot()

            img = images[i]

            tag = 0
            for x in myroot.findall('object'):
                name = x.find('name')

                if (class_name[0] in name.text or class_name[1] in name.text or class_name[2] in name.text):
                    xmin, ymin, xmax, ymax = self.getXYcoorinates(x, tag_name)  # xy-coordinates ground truth
                    tag += 1
                    cropped_img = img[ymin:ymax, xmin:xmax]
                    images_positive.append(cropped_img)
                elif "building" in name.text:
                    xmin, ymin, xmax, ymax = self.getXYcoorinates(x, tag_name)
                    tag += 1
                    cropped_img = img[ymin:ymax, xmin:xmax]
                    images_negative.append(cropped_img)
                else:
                    pass

        return images_positive, images_negative


    def rescaleTrainImages(self, train_images, threshold_1, threshold_2, threshold_3):
        """
        Rescale images of the dataset to approximately same size
        :param train_images: list of images to rescale
        :param threshold_1: the upper height threshold
        :param threshold_2: the middle height threshold
        :param threshold_3: the lower height threshold
        :return: list of rescaled images
        """
        train_images_ship = []

        for ind in range(len(train_images)):
            img_shape = np.shape(train_images[ind])  # (0:height, 1:width)
            height = img_shape[0]  # y
            width = img_shape[1]  # x


            if height > threshold_1:
                scale = threshold_1 / height
                new_width = int(width * scale)
                img_downsize = cv2.resize(train_images[ind], (new_width, threshold_1))
                train_images_ship.append(img_downsize)
            elif threshold_3 < height < threshold_2:
                scale = threshold_2 / height
                new_width = int(width * scale)
                img_upscale = cv2.resize(train_images[ind], (new_width, threshold_2))
                train_images_ship.append(img_upscale)
            elif height < threshold_3:
                pass
            else:
                train_images_ship.append(train_images[ind])
        return train_images_ship

    def rescaleBoundingBox(self, xmin, ymin, xmax, ymax, scale):
        """
        Rescale bounding boxes after the rescaled image
        :param xmin: start x-coordinate
        :param ymin: start y-coordinate
        :param xmax: end x-coordinate
        :param ymax: end y-coordinate
        :param scale: the scale factor
        :return: coordinate of new  bounding box
        """
        x_new = int(np.round(xmin * scale))
        y_new = int(np.round(ymin * scale))
        x_end_new = int(np.round(xmax * scale))
        y_end_new = int(np.round(ymax * scale))
        coordinate = [x_new, y_new, x_end_new, y_end_new]
        return coordinate