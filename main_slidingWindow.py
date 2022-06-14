"""
Classification of two classes: ship and non-ship using BoVW with sliding window approach
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import distance
from skimage.feature import hog
from sklearn.cluster import KMeans
import imutils
from xml.etree import ElementTree as et
from image_processing import ImageProcessing



def pyramid(image, scale, minSize):
    """
    Run a image pyramid over sliding window
    :param image: the image to downsample
    :param scale: the scale to downsample image
    :param minSize: minimum size of window
    """
    yield image

    # Keep looping over the pyramid
    while True:
        # Compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSizeX, stepSizeY, windowSize):
    """
    Sliding window function
    :param image: The image to iterate through
    :param stepSize: How many pixels to "skip" in both the (x, y) direction.
    :param windowSize: Width and height (in terms of pixels) of the window we are going to extract from out image
    """
    # Slide a window across the image
    for y in range(0, image.shape[0], stepSizeY):
        for x in range(0, image.shape[1], stepSizeX):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def getImagePatches(train_images, threshold, save, save_dir):
    """
    Extract image patches from the train images
    :param train_images: list of images to do feature extraction of
    :param threshold: variance threshold to determine what patches to be extracted
    :param save: boolean to determine to save image
    :param save_dir: directory to save image
    :return: list of extracted image patches, coordinates, variance value for all patches
    """
    image_patches = []
    variance = []
    # Use sliding window to get patches
    image_nr = -1
    for input_img in train_images:
        image_nr += 1

        window_width = 64
        window_height = 64
        stride_x = int(window_width * 0.5)
        stride_y = int(window_height * 0.5)

        # Loop over the image pyramid
        loop = 0
        for resized in pyramid(input_img, scale=1.5, minSize=(100, 100)):
            loop += 1
            # Create output image to store patches
            img_shape = np.shape(resized)  # (0:height, 1:width)
            output_img = np.full(shape=(img_shape[0], img_shape[1]), fill_value=0, dtype=np.int)
            # Sliding window algorithm - Create patches for each window

            for (x, y, window) in sliding_window(resized, stepSizeX=stride_x, stepSizeY=stride_y,
                                                 windowSize=(window_width, window_height)):
                # if our window does not meet our desired window size, ignore it
                if window.shape[0] != window_height or window.shape[1] != window_width:
                    continue

                window_patch_gray = resized[y:y + window_height, x:x + window_width]
                # Compute variance
                intensity = ndimage.variance(window_patch_gray)
                variance.append(intensity)

                var_threshold = threshold
                if intensity > var_threshold:
                    # Store image patches
                    image_patches.append(window_patch_gray)
                    # Add image patch to output_img in the same position it was taken out
                    output_img[y:y + window_height, x:x + window_width] = window_patch_gray
                    if save is True:  # store output image with extracted image patches
                        img_name = "image_" + str(image_nr) + "_" + str(loop) + ".jpg"
                        processing.saveImageToFolder(output_img, img_name, save_dir)

    return image_patches, variance


def getFeaturesSlidingWindow(train_images, threshold, show_hist, HoG_param, save_bool, save_dir):
    """
    Get image patches and compute descriptor for each patch
    :param train_images: the images to do feature extraction on
    :param threshold: variance threshold
    :param show_hist: boolean to determine to show variance histogram
    :param HoG_param: parameter values for HoG descriptor
    :param save_bool: boolean to determine to save output image consisting of extracted image patches
    :param save_dir: directory to save output image
    :return: list of extracted image_patches, list of descriptors
    """
    image_patches, var_arr = getImagePatches(train_images, threshold, save_bool, save_dir)

    # Check variances from each image patch
    plt.style.use('seaborn-deep')
    if show_hist is True:
        plt.hist(var_arr, rwidth=0.9)
        plt.grid()
        plt.title("Histogram of Variance")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.axvline(x=threshold, ymin=0, ymax=250000, color='red', linestyle='dashed', linewidth=1)
        plt.text(x=threshold + 20, y=600000, s='Threshold', fontsize=10)
        plt.show()

    # Compute descriptor for each image patch
    descriptors = []
    for patch in image_patches:
        des_hog, hog_image = hog(patch, orientations=HoG_param[0], pixels_per_cell=HoG_param[1],
                                 cells_per_block=HoG_param[2], visualize=True, multichannel=False)
        descriptors.append(des_hog)
    return image_patches, descriptors


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


def intersection_over_union(box_A, box_B):
    """
    Compute intersection over union
    :param box_A: predicted bounding box
    :param box_B: ground truth bounding box
    :return: the IoU
    """
    # Determine the xy-coordinates for the intersection rectangle
    x_A = max(box_A[0], box_B[0])
    y_A = max(box_A[1], box_B[1])
    x_B = min(box_A[2], box_B[2])
    y_B = min(box_A[3], box_B[3])

    # Compute the area of intersection rectangle
    interArea = max(0, x_B - x_A + 1) * max(0, y_B - y_A + 1)
    # compute the area of both the prediction and ground truth bounding boxes
    boxA_area = (box_A[2] - box_A[0] + 1) * (box_A[3] - box_A[1] + 1)
    boxB_area = (box_B[2] - box_B[0] + 1) * (box_B[3] - box_B[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxA_area + boxB_area - interArea)
    return iou


if __name__ == '__main__':

    # Parameter values - Training
    # Get features sliding window:
    variance_threshold_train = 1500
    save_images_to_folder = False
    show_variance_histogram = False
    hog_param = [9, (8, 8), (2, 2)]  # HoG parameters
    # Filter vocabulary
    diff_threshold = 4.5
    # Vocabulary generation
    N_clusters_ship = 1000  # Number of clusters for ship class
    N_clusters_nonship = 1000  # Number of clusters for non-ship class
    top_N_ship = 20  # Top N descriptors to cluster centre
    top_N_nonship = 20
    init_method = 'k-means++'

    # Parameter values - Testing
    variance_threshold_test = 1500
    T_classification = 0.1
    save_folder_test = "results/classification/multi/k1000_500/test2"

    processing = ImageProcessing()

    print("*** START TRAINING ***")
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
    ship_patches_train, descriptors_ship_train = getFeaturesSlidingWindow(train_ship, variance_threshold_train,
                                                                          show_variance_histogram, hog_param,
                                                                          save_images_to_folder,
                                                                          'results/output_image_stride_ship')
    print("Features extracted for ship")

    nonship_patches_train, descriptors_nonship_train = getFeaturesSlidingWindow(train_nonship, variance_threshold_train,
                                                                                show_variance_histogram, hog_param,
                                                                                save_images_to_folder,
                                                                                'results/output_image_stride_nonship')
    print("Features extrated for non-ship", '\n')

    print("Number of ship patches:", len(ship_patches_train), ",", len(nonship_patches_train))
    print("Number of ship descriptors:", len(descriptors_ship_train), ",", len(descriptors_nonship_train))

    print("Features extracted from training and testing completed", '\n')

    ship_train = np.float32(descriptors_ship_train)
    nonship_train = np.float32(descriptors_nonship_train)

    # Match descriptors in ship and non-ship and remove descriptors that are similar
    flann = cv2.FlannBasedMatcher()
    matches_ship = flann.knnMatch(ship_train, nonship_train, k=1)
    matches_nonship = flann.knnMatch(nonship_train, ship_train, k=1)

    train_ship_descriptors = []
    train_nonship_descriptors = []
    image_patches_ship = []
    image_patches_nonship = []
    for match_ship in matches_ship:
        if match_ship[0].distance <= diff_threshold:
            pass
        else:
            train_ship_descriptors.append(descriptors_ship_train[match_ship[0].queryIdx])
            image_patches_ship.append(ship_patches_train[match_ship[0].queryIdx])

    for match_nonship in matches_nonship:
        if match_nonship[0].distance <= diff_threshold:
            pass
        else:
            train_nonship_descriptors.append(descriptors_nonship_train[match_nonship[0].queryIdx])
            image_patches_nonship.append(nonship_patches_train[match_nonship[0].queryIdx])

    print("Ship descriptors after filtering:", len(train_ship_descriptors))
    print("Non-ship descriptors after filtering:", len(train_nonship_descriptors))
    print("Number of image patches:", len(image_patches_ship), ",", len(image_patches_nonship))

    # Generate vocabulary for both ship and non-ship class
    descriptor_list_ship, centroids_ship, visual_words_ship = vocabularyGeneration(N_clusters_ship, init_method,
                                                                                   train_ship_descriptors, top_N_ship)

    descriptor_list_nonship, centroids_nonship, visual_words_nonship = vocabularyGeneration(N_clusters_nonship,
                                                                                            init_method,
                                                                                            train_nonship_descriptors,
                                                                                            top_N_nonship)

    print("Vocabulary generation completed")

    print("Number of clusters:", len(centroids_ship), ",", len(centroids_nonship))

    # Visualize vocabulary
    visualizeVocabulary(visual_words_ship, image_patches_ship, top_N_ship, nr_random_clusters=10, map='gray')

    visualizeVocabulary(visual_words_nonship, image_patches_nonship, top_N_nonship, nr_random_clusters=10, map='gray')

    print("")
    print("")
    print("*** START TESTING ***")

    test_images_path, test_annotation_path = processing.getFilePath('test_data')

    # Get ground truth bounding boxes
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

    # Read images from folder
    test_images = processing.readImagesInFolder(test_images_path)

    # Resize test images with corresponding ground truth
    resized_test, new_ground_truth = cropTestImagesAndGroundTruth(test_images, ground_truth, 650, 300, 100)

    # Extract image patches using slinding window
    confidence_score_list = []
    IoU_list = []
    img_ind = -1
    for image in resized_test:
        img_ind += 1
        img_shape = np.shape(image)  # (0:height, 1:width)
        window_width = 64
        window_height = 64
        stride_x = int(window_width * 0.5)
        stride_y = int(window_height * 0.5)

        image_patches = []
        descriptor_list = []
        coordinate_list = []
        for (x, y, window) in sliding_window(image, stepSizeX=stride_x, stepSizeY=stride_y,
                                             windowSize=(window_width, window_height)):
            # if our window does not meet our desired window size, ignore it
            if window.shape[0] != window_height or window.shape[1] != window_width:
                continue

            window_patch_gray = image[y:y + window_height, x:x + window_width]

            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 4)

            intensity = ndimage.variance(window_patch_gray)
            coordinates = []
            if intensity > variance_threshold_test:
                image_patches.append(window_patch_gray)
                start_coord = (x, y)
                end_coord = (x + window_width, y + window_height)
                coordinates.append(start_coord)
                coordinates.append(end_coord)
                coordinate_list.append(coordinates)

                # Compute descriptor
                des_hog, hog_image = hog(window_patch_gray, orientations=hog_param[0], pixels_per_cell=hog_param[1],
                                         cells_per_block=hog_param[2], visualize=True, multichannel=False)
                descriptor_list.append(des_hog)

        print("")
        print("Number of patches:", np.shape(image_patches))
        print("Number of descriptors:", np.shape(descriptor_list))
        print("Number of coordinates:", np.shape(coordinate_list))

        print("Size centroids ship:", np.shape(centroids_ship))
        print("Size centroids non-ship:", np.shape(centroids_nonship))

        descriptor = np.float32(descriptor_list)
        centre_ship = np.float32(centroids_ship)
        centre_nonship = np.float32(centroids_nonship)

        # Match test descriptors with cluster centres of both vocabularies
        flann = cv2.FlannBasedMatcher()

        knn_matches_ship = flann.knnMatch(queryDescriptors=descriptor, trainDescriptors=centre_ship, k=2)
        knn_matches_nonship = flann.knnMatch(queryDescriptors=descriptor, trainDescriptors=centre_nonship, k=2)

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

        # Create output images
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
            x = coordinates[0][0]
            y = coordinates[0][1]
            x_end = coordinates[1][0]
            y_end = coordinates[1][1]

            dist_t = best_matches_ship[descriptor_index].distance
            dist_b = best_matches_nonship[descriptor_index].distance

            diff = np.abs(dist_t - dist_b)

            # Visualize image patches on image
            if (dist_t < dist_b) and (diff > T_classification):
                # SHIP
                count_pos += 1
                x_array_positive.append(x)
                y_array_positive.append(y)
                x_array_positive.append(x_end)
                y_array_positive.append(y_end)
                output_image_test[y:y_end, x:x_end] = image_patches[descriptor_index]
                cv2.rectangle(copy_test, (x, y), (x_end, y_end), (0, 255, 0), 1)
            elif (dist_b < dist_t) and (diff > T_classification):
                # NON-SHIP
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

        # Save output image
        # cv2.imwrite(os.path.join(save_folder_test, "testimage_" + str(img_ind) + ".png"), copy_test)
        # cv2.waitKey(0)

        plt.imshow(output_image_test, cmap='gray')
        plt.show()

        # Save output image
        # cv2.imwrite(os.path.join(save_folder_test, "outputimage_" + str(img_ind) + ".png"), output_image_test)
        # cv2.waitKey(0)

        print("")
        print("")

        # Classify test image
        nr_patches_ship = count_pos
        print("Total number of positive patches:", nr_patches_ship)
        nr_patches_other = count_neg
        print("Total number of negative patches:", nr_patches_other)
        total_count = count_pos + count_neg

        # Ground truth - from .xml file
        x_truth = new_ground_truth[img_ind][0]
        y_truth = new_ground_truth[img_ind][1]
        x_end_truth = new_ground_truth[img_ind][2]
        y_end_truth = new_ground_truth[img_ind][3]
        ground_truth_box = cv2.rectangle(copy_test_predicted, (x_truth, y_truth), (x_end_truth, y_end_truth),
                                         (255, 0, 0), 2)
        ground_truth = (x_truth, y_truth, x_end_truth, y_end_truth)

        # Classify image
        # If more patches are classified as ship, then the object is classified as ship
        if nr_patches_ship > nr_patches_other:
            label = "Ship"
            # Compute score - How sure is the classification?
            confidence_score = "{:.2f}".format(count_pos / total_count)
            confidence_score_list.append(confidence_score)
            # Generate ROI with corresponding color - GREEN
            x_pred = min(x_array_positive)
            y_pred = min(y_array_positive)
            x_end_pred = max(x_array_positive)
            y_end_pred = max(y_array_positive)
            predicted_box = cv2.rectangle(copy_test_predicted, (x_pred, y_pred), (x_end_pred, y_end_pred), (0, 255, 0),
                                          2)
            cv2.putText(copy_test_predicted, label + ", " + str(confidence_score), (x_pred + 10, y_pred + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Intersection over union (one image) - How sure is localization
            IoU = intersection_over_union((x_pred, y_pred, x_end_pred, y_end_pred), ground_truth)
        else:
            label = "Non-ship"
            confidence_score = "{:.2f}".format(count_neg / total_count)
            confidence_score_list.append(confidence_score)
            x = min(x_array_negative)
            y = min(y_array_negative)
            x_end = max(x_array_negative)
            y_end = max(y_array_negative)
            predicted_box = cv2.rectangle(copy_test_predicted, (x, y), (x_end, y_end), (0, 0, 255), 2)
            cv2.putText(copy_test_predicted, label + ", " + str(confidence_score), (x + 10, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Intersection over union (one image)
            IoU = intersection_over_union((x, y, x_end, y_end), ground_truth)

        # Find the object
        cv2.imshow("test image", copy_test_predicted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save output image
        # cv2.imwrite(os.path.join(save_folder_test, "outputimagepredicted_" + str(img_ind) + ".png"), copy_test_predicted)
        # cv2.waitKey(0)






