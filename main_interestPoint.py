"""
Classification of two classes: ship and non-ship using BoVW with interest point approach
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import distance
from sklearn.cluster import KMeans
from xml.etree import ElementTree as et
from image_processing import ImageProcessing


def getImagePatchesFromKp(image_list, kp_list):
    """
    Get image patch from interest point
    :param image_list: list of images
    :param kp_list: list of keypoints
    :return: image patches and corresponding coordinates
    """
    x = 0
    y = 1
    patches = []
    coordinates = []
    for img_ind in range(len(image_list)):
        img = image_list[img_ind]
        img_shape = np.shape(img)
        x_start = 0
        y_start = 0
        x_end = img_shape[1]
        y_end = img_shape[0]

        coord = []
        for kp_ind in range(len(kp_list[img_ind])):  # index for all kp in one image
            ymin = kp_list[img_ind][kp_ind].pt[y] - (kp_list[img_ind][kp_ind].size / 2)
            ymax = kp_list[img_ind][kp_ind].pt[y] + (kp_list[img_ind][kp_ind].size / 2)
            xmin = kp_list[img_ind][kp_ind].pt[x] - (kp_list[img_ind][kp_ind].size / 2)
            xmax = kp_list[img_ind][kp_ind].pt[x] + (kp_list[img_ind][kp_ind].size / 2)
            # Check coordinates
            if (xmin < x_start or ymin < y_start or xmax > x_end or ymax > y_end):
                #print("Image patch outside of image window. Patch ignored")
                pass
            else:
                image_patch = img[int(float(ymin)):int(float(ymax)), int(float(xmin)):int(float(xmax))]
                patches.append(image_patch)
                coord.append((xmin, ymin))
                coord.append((xmax, ymax))
                coordinates.append(coord)
    return patches, coordinates



def getFeaturesInterestPoints(images):
    """
    Extract features using interest point extractor
    :param images: image to extract patches from
    :param detector: SIFT de
    :return:
    """
    # Define feature extractor
    x = 0
    y = 1
    extractor = cv2.SIFT_create()

    des_list = []
    patch_arr = []
    size_threshold = 10
    tag = 0
    for img in images:
        img_shape = np.shape(img)
        x_start = 0
        y_start = 0
        x_end = img_shape[1]
        y_end = img_shape[0]

        output_img = np.full(shape=(img_shape[0], img_shape[1]), fill_value=0, dtype=np.int)

        tag += 1
        keypoint = extractor.detect(img, None)
        # img_kp = cv2.drawKeypoints(img, keypoint, 0, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(img_kp, cmap='gray')
        # plt.show()
        kp_list = []
        if len(keypoint) != 0:
            for kp_ind in range(len(keypoint)):
                if (keypoint[kp_ind].size >= size_threshold):
                    ymin = keypoint[kp_ind].pt[y] - (keypoint[kp_ind].size / 2)
                    ymax = keypoint[kp_ind].pt[y] + (keypoint[kp_ind].size / 2)
                    xmin = keypoint[kp_ind].pt[x] - (keypoint[kp_ind].size / 2)
                    xmax = keypoint[kp_ind].pt[x] + (keypoint[kp_ind].size / 2)
                    if (xmin < x_start or ymin < y_start or xmax > x_end or ymax > y_end):
                        # print("Image patch outside of image window. Patch ignored")
                        pass
                    else:
                        image_patch = img[int(float(ymin)):int(float(ymax)), int(float(xmin)):int(float(xmax))]
                        # output_img[int(ymin):int(ymax), int(xmin):int(xmax)] = image_patch
                        # img_name = "image_SIFT_" + str(tag) + ".jpg"
                        # saveImageToFolder(output_img, img_name, 'results')
                        patch_arr.append(image_patch)
                        kp_list.append(keypoint[kp_ind])
            if len(kp_list) != 0:
                kp, descriptor = extractor.compute(img, kp_list)
                des_list.append(descriptor)
        else:
            pass

    # Flatten descriptor list
    descriptor_arr = []
    for i in range(len(des_list)):
        for ind in range(len(des_list[i])):
            descriptor_arr.append(des_list[i][ind])

    return patch_arr, descriptor_arr


def computeKmeans(n_dictionary, init, des):
    """
    Define k-means algorithm and fit descriptors
    :param n_dictionary: Number of clusters
    :param init: initialization method. "k-means++" is prefered
    :param des: descriptor list to cluster
    :return: k-means model
    """
    kmeans = KMeans(n_clusters=n_dictionary,
                          verbose=False,
                          init=init,
                          random_state=1,
                          n_init=3)
    # fit the model
    kmeans.fit(des)
    return kmeans


def printDescriptorIndexFromKey(key, descriptors, cluster_labels):
    """
    Get indexes of descriptors in one cluster
    :param key: the cluster index
    :param descriptors: list of descriptors
    :param cluster_labels: labels of descriptors
    :return: list of all descriptor indexes in the given cluster
    """
    index_kmeans = {}
    for index in range(len(descriptors)):
        index_kmeans[index] = str(cluster_labels[index])

    index_des = {}
    for pair in index_kmeans.items():
        if pair[1] not in index_des.keys():
            index_des[pair[1]] = []
        index_des[pair[1]].append(pair[0])
    return index_des.get(key)


def getVocabulary(cluster_descriptor, codeword, N):
    """
    Get index of the top N similar descriptors to the cluster centre
    :param cluster_descriptor: list with descriptors indexes for each cluster
    :param codeword: list with cluster centres
    :param N: The top N most similar descriptors to cluster center
    :return: list of indexes of the N similar descriptors
    """
    words_index = []
    tag = 0
    for key in range(len(cluster_descriptor)):
        tag += 1

        # computes distances for cluster i
        distances = []
        for ind in range(len(cluster_descriptor[key])):
            # Compute Euclidean distance
            euclidean = distance.euclidean(codeword[key], cluster_descriptor[key][ind])
            distances.append(euclidean)

        # Get index for top N elements
        indx = np.argpartition(distances, N)[:N]
        words_index.append(indx)
    return words_index


def vocabularyGeneration(no_clusters, init_method, descriptors, N):
    """
    Generate vocabulary
    :param no_clusters: Size of vocabulary
    :param init_method: initialization method for k-means (k-means++)
    :param descriptors: list of descriptors to cluster
    :param N: number of similar descriptors to cluster centres
    :return: list of top N descriptor index for each cluster, cluster centres
    """
    # Define a KMeans clustering model
    kmeans_model = computeKmeans(no_clusters, init=init_method, des=descriptors)
    labels = kmeans_model.labels_  # Label for each descriptor of the training set
    print("Shape labels:", np.shape(labels))

    centroid_list = kmeans_model.cluster_centers_
    print("Shape of array with centroids:", np.shape(centroid_list))

    descriptor_in_cluster = []
    for key in range(no_clusters):
        # Get index of descriptors
        index_descriptor = printDescriptorIndexFromKey(str(key), descriptors, labels)
        des = []
        for ind in index_descriptor:
            des.append(descriptors[ind])
        descriptor_in_cluster.append(des)

    # Remove clusters with less than top_N
    new_descriptor_in_cluster_list = []
    centroids = []
    for index in range(no_clusters):
        list = descriptor_in_cluster[index]
        if len(list) > N:
            new_descriptor_in_cluster_list.append(list)
            centroids.append(centroid_list[index])

    print("Length cluster list before filtering:", len(descriptor_in_cluster))
    print("Length cluster list after filtering:", len(new_descriptor_in_cluster_list))
    print("Length centroid list after filtering:", len(centroids))

    words_index = getVocabulary(new_descriptor_in_cluster_list, centroids, N)
    return new_descriptor_in_cluster_list, centroids, words_index


def visualizeVocabulary(words, image_patches, N, nr_random_clusters, map):
    """
    Visualize the vocabulary
    :param words: centroid list
    :param image_patches: image patches from training set
    :param N: number of similar descriptors to cluster centres
    :param nr_random_clusters: number of clusters to visualize
    :param map: the cmap of matplotlib. Preferable as "Gray"
    """
    # Visualize top N patches from 10 random clusters
    visual_words = []
    for cluster in words:
        visual_words_patch = []
        for index in cluster:
            patch = image_patches[index]
            visual_words_patch.append(patch)
        visual_words.append(visual_words_patch)
    flattened = [val for sublist in visual_words for val in sublist]

    for i in range(nr_random_clusters * N):
        plt.subplot(nr_random_clusters, N, i + 1), plt.imshow(flattened[i], cmap=map)
        plt.xticks([]), plt.yticks([])
    plt.show()



def cropTestImagesAndGroundTruth(test_images, box_coordinates, threshold_1, threshold_2, threshold_3):
    """
    Resize image and corresponding bounding boxes. Used during testing phase
    :param test_images: the images to rescale
    :param box_coordinates: coordinates of original bounding boxes
    :param threshold_1: upper threshold
    :param threshold_2: middle threshold
    :param threshold_3: lower threshold
    :return: the rescaled image with corresponding bounding box
    """
    images = []
    new_box_coordinates = []

    for ind in range(len(test_images)):
        img_shape = np.shape(test_images[ind])  # (0:height, 1:width)
        height = img_shape[0]  # y
        width = img_shape[1]  # x

        # Original bounding box coordinates
        x = box_coordinates[ind][0]
        y = box_coordinates[ind][1]
        x_end = box_coordinates[ind][2]
        y_end = box_coordinates[ind][3]

        if height > threshold_1:
            scale = threshold_1 / height
            new_width = int(width * scale)
            img_downsize = cv2.resize(test_images[ind], (new_width, threshold_1))
            images.append(img_downsize)

            # Downsize bounding box
            coord = processing.rescaleBoundingBox(x, y, x_end, y_end, scale)
            new_box_coordinates.append(coord)
        elif threshold_3 < height < threshold_2:
            scale = threshold_2 / height
            new_width = int(width * scale)
            img_upscale = cv2.resize(test_images[ind], (new_width, threshold_2))
            images.append(img_upscale)

            # Upsample bounding box
            coord = processing.rescaleBoundingBox(x, y, x_end, y_end, scale)
            new_box_coordinates.append(coord)
        elif height < threshold_3:
            pass
        else:
            images.append(test_images[ind])
            coord = [x, y, x_end, y_end]
            new_box_coordinates.append(coord)
    return images, new_box_coordinates



if __name__ == '__main__':

    processing = ImageProcessing()

    print("*** START TRAINING ***")
    T_classification = 20

    train_images_path, train_annotation_path = processing.getFilePath('train_data')
    print("Number of images", len(train_images_path))
    print("Number of annotation files", len(train_annotation_path))

    # Read images
    train_images = processing.readImagesInFolder(train_images_path)


    # Crop out objects based on ground truth boxes
    cropped_images_ship, cropped_images_nonship = processing.cropTrainImages(train_images, train_annotation_path)
    print("Number of ship images in training array:", len(cropped_images_ship))
    print("Number of non-ship images in training array:", len(cropped_images_nonship), '\n')

    # Rescale images with corresponding ground truth box
    train_ship = processing.rescaleTrainImages(cropped_images_ship, threshold_1=650, threshold_2=350,
                                    threshold_3=150)
    train_nonship = processing.rescaleTrainImages(cropped_images_nonship, threshold_1=650, threshold_2=350,
                                       threshold_3=150)

    print("(After rescaling) Number of ship images in training array:", len(train_ship))
    print("(After rescaling) Number of non-ship images in training array:", len(train_nonship), '\n')

    # Extract features from both ship and non-ship category using sliding window
    ship_patches_train, descriptors_ship_train = getFeaturesInterestPoints(train_ship)
    print("Number of patches train:", len(ship_patches_train))
    print("Number of descriptors train:", len(descriptors_ship_train))

    nonship_patches_train, descriptors_nonship_train = getFeaturesInterestPoints(train_nonship)
    print("Number of patches:", len(ship_patches_train), "(ship) ,", len(nonship_patches_train), "(non-ship)")
    print("Number of descriptors:", len(descriptors_ship_train), "(ship) ,", len(descriptors_nonship_train),
          "(non-ship)")
    print("Features extracted from training and testing completed", '\n')


    # Generate vocabulary for both ship and non-ship class
    print("Vocabulary generation")
    N_clusters_ship = 1000  # Number of clusters for ship class
    N_clusters_nonship = 1000
    top_N = 20  # Top N descriptors to cluster centre
    descriptor_list_ship, centroids_ship, visual_words_ship = vocabularyGeneration(N_clusters_ship, "k-means++",
                                                                               descriptors_ship_train, top_N)
    visualizeVocabulary(visual_words_ship, ship_patches_train, top_N, nr_random_clusters=10, map='gray')
    descriptor_list_nonship, centroids_nonship, visual_words_nonship = vocabularyGeneration(N_clusters_nonship, "k-means++",
                                                                                        descriptors_nonship_train, top_N)
    visualizeVocabulary(visual_words_ship, nonship_patches_train, top_N, nr_random_clusters=10, map='gray')


    print("Vocabulary generation completed", '\n')

    print("")
    print("")
    print("*** START TESTING ***")

    test_images_path, test_annotation_path = processing.getFilePath('test_data')

    # Get ground truth coordinates
    ground_truth = []
    for i in range(len(test_images_path)):
        tree = et.parse(test_annotation_path[i])
        myroot = tree.getroot()

        tag_name = ['xmin', 'ymin', 'xmax', 'ymax']

        for x in myroot.findall('object'):
            xmin, ymin, xmax, ymax = processing.getXYcoorinates(x, tag_name)
            coord = [xmin, ymin, xmax, ymax]
            # print(coord)
            # Resize bounding box after input image
            ground_truth.append(coord)

    # Read images
    test_images = processing.readImagesInFolder(test_images_path)

    # Resize test images with corresponding ground truth
    resized_test, new_ground_truth = cropTestImagesAndGroundTruth(test_images, ground_truth, 650, 300, 100)

    img_ind = -1
    kp_list = []
    des_list = []
    images_test = []
    image_patches = []
    threshold = 10
    for image in resized_test:
        # Image info
        img_shape = np.shape(image)
        x_start = 0
        y_start = 0
        x_end = img_shape[1]
        y_end = img_shape[0]


        img_ind += 1
        print("Extractor selected for test: SIFT")
        extractor = cv2.SIFT_create()
        keypoint, descriptor = extractor.detectAndCompute(image, None)
        if descriptor is not None:
            images_test.append(image)
            kp_list.append(keypoint)
            des_list.append(descriptor)

        coordinate_list = []
        descriptor_list = []
        x = 0
        y = 1
        for kp_ind in range(len(keypoint)):
            des = descriptor[kp_ind]

            ymin = kp_list[img_ind][kp_ind].pt[y] - (kp_list[img_ind][kp_ind].size / 2)
            ymax = kp_list[img_ind][kp_ind].pt[y] + (kp_list[img_ind][kp_ind].size / 2)
            xmin = kp_list[img_ind][kp_ind].pt[x] - (kp_list[img_ind][kp_ind].size / 2)
            xmax = kp_list[img_ind][kp_ind].pt[x] + (kp_list[img_ind][kp_ind].size / 2)
            if (xmin < x_start or ymin < y_start or xmax > x_end or ymax > y_end):
                pass
            else:
                image_patch = image[int(float(ymin)):int(float(ymax)), int(float(xmin)):int(float(xmax))]
                image_patches.append(image_patch)
                coordinate_list.append((xmin, ymin, xmax, ymax))
                descriptor_list.append(des)

        print("Number of patches:", np.shape(image_patches))
        print("Number of descriptors:", np.shape(descriptor_list))
        print("Number of coordinates:", np.shape(coordinate_list))

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
        descriptors = np.float32(descriptor_list)
        centre_ship = np.float32(centroids_ship)
        centre_nonship = np.float32(centroids_nonship)

        print("shape descriptors from one image:", np.shape(descriptors))
        print("shape centroids ship:", np.shape(centre_ship))
        print("shape centroids non-ship:", np.shape(centre_nonship))
        print("")

        # Find the k best matches for each descriptor from a train set
        knn_matches_ship = matcher.knnMatch(queryDescriptors=descriptors, trainDescriptors=centre_ship, k=2)
        knn_matches_nonship = matcher.knnMatch(queryDescriptors=descriptors, trainDescriptors=centre_nonship, k=2)

        # Ratio test from D.Lowes paper
        best_matches_ship = []
        test_image_patches = []
        for match1, match2 in knn_matches_ship:
            if match1.distance < match2.distance:
                best_matches_ship.append(match1)
            else:
                best_matches_ship.append(match2)

        best_matches_nonship = []
        for match1, match2 in knn_matches_nonship:
            if match1.distance < match2.distance:
                best_matches_nonship.append(match1)
            else:
                best_matches_nonship.append(match2)


        output_image_test = np.full(shape=(img_shape[0], img_shape[1]), fill_value=0, dtype=np.int)
        copy_test = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        copy_test_predicted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Count number of positive and negative patches
        count_pos = 0
        count_neg = 0
        x_array_positive = []
        x_array_negative = []
        y_array_positive = []
        y_array_negative = []
        for descriptor_index in range(len(descriptor_list)):
            coordinates = coordinate_list[descriptor_index]
            x = int(coordinates[0])
            y = int(coordinates[1])
            x_end = int(coordinates[2])
            y_end = int(coordinates[3])
            # print("Coordinates:", (x, y, x_end, y_end))

            dist_t = best_matches_ship[descriptor_index].distance  # distansen til ship vocabulary for en descriptor
            dist_b = best_matches_nonship[descriptor_index].distance
            diff = np.abs(dist_t - dist_b)
            print(diff)

            # Visualize image patches on image
            if (dist_t < dist_b) and (diff > T_classification):
                # print("descriptor is ship")
                count_pos += 1
                x_array_positive.append(x)
                y_array_positive.append(y)
                x_array_positive.append(x_end)
                y_array_positive.append(y_end)
                output_image_test[y:y_end, x:x_end] = image_patches[descriptor_index]
                cv2.rectangle(copy_test, (x, y), (x_end, y_end), (0, 255, 0), 1)
            elif (dist_b < dist_t) and (diff > T_classification):
                # print("descriptor is non-ship")
                count_neg += 1
                x_array_negative.append(x)
                y_array_negative.append(y)
                x_array_negative.append(x_end)
                y_array_negative.append(y_end)
                cv2.rectangle(copy_test, (x, y), (x_end, y_end), (0, 0, 255), 1)
            else:
                # NOT CLASSIFIED
                cv2.rectangle(copy_test, (x, y), (x_end, y_end), (217, 77, 30), 1)
        # Find the object
        cv2.imshow("test image", copy_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        plt.imshow(output_image_test, cmap='gray')
        plt.show()







