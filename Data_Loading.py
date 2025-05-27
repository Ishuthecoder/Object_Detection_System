import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import cv2

# VOC Classes list (20 classes)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

class VOCDatasetLoader:
    def __init__(self, voc_root, image_size=(300, 300)):
        """
        voc_root: Root folder path of VOC dataset (folder containing Annotations, JPEGImages etc.)
        image_size: Target size for resizing images (width, height)
        """
        self.voc_root = voc_root
        self.image_size = image_size
        self.annotations_dir = os.path.join(voc_root, 'Annotations')
        self.images_dir = os.path.join(voc_root, 'JPEGImages')

        # List of annotation xml files
        self.annotation_files = [os.path.join(self.annotations_dir, f) 
                                 for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]

    def _parse_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image filename
        filename = root.find('filename').text
        img_path = os.path.join(self.images_dir, filename)

        # Read the image to get original size
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found or corrupted.")
        orig_height, orig_width = image.shape[:2]

        boxes = []
        labels = []

        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in VOC_CLASSES:
                continue

            cls_id = VOC_CLASSES.index(cls_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / orig_width
            ymin = float(bndbox.find('ymin').text) / orig_height
            xmax = float(bndbox.find('xmax').text) / orig_width
            ymax = float(bndbox.find('ymax').text) / orig_height

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)

        return img_path, boxes, labels

    def _load_image_and_labels(self, xml_file):
        img_path, boxes, labels = self._parse_annotation(xml_file)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size) / 255.0  # Normalize pixel values [0,1]

        # Convert boxes and labels to tensors
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        return image, boxes, labels

    def get_tf_dataset(self, batch_size=8, shuffle=True):
        # Create TensorFlow Dataset from annotation files list
        dataset = tf.data.Dataset.from_tensor_slices(self.annotation_files)

        def _load_data(xml_path):
            image, boxes, labels = tf.py_function(
                func=lambda x: self._load_image_and_labels(x.numpy().decode()),
                inp=[xml_path],
                Tout=(tf.float32, tf.float32, tf.int32)
            )
            # Set shapes for TF graph
            image.set_shape([*self.image_size, 3])
            boxes.set_shape([None, 4])
            labels.set_shape([None])
            return image, boxes, labels

        dataset = dataset.map(_load_data, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.padded_batch(batch_size, 
                                       padded_shapes=([*self.image_size, 3], [None, 4], [None]),
                                       padding_values=(0.0, 0.0, -1))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


# Example usage:
if __name__ == "__main__":
    voc_path = r"C:\Users\dubey\Downloads\archive\VOC2012"  # Change to your VOC dataset root folder path
    dataset_loader = VOCDatasetLoader(voc_path, image_size=(300,300))

    train_dataset = dataset_loader.get_tf_dataset(batch_size=4)

    for images, boxes, labels in train_dataset.take(1):
        print("Batch images shape:", images.shape)
        print("Batch boxes shape:", boxes.shape)
        print("Batch labels shape:", labels.shape)
        print("First image boxes:", boxes[0].numpy())
        print("First image labels:", labels[0].numpy())

