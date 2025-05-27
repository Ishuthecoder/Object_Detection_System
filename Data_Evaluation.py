import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, average_precision_score
import xml.etree.ElementTree as ET

# === Configuration ===
IMG_SIZE = 300
MAX_BOXES = 10
BATCH_SIZE = 8

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

NUM_CLASSES = len(VOC_CLASSES)

class_to_id = {cls: i for i, cls in enumerate(VOC_CLASSES)}

VOC_ROOT = r"C:\Users\dubey\Downloads\archive\VOC2012"
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
JPEGIMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")
IMAGESETS_MAIN = os.path.join(VOC_ROOT, "ImageSets", "Main")

# === Utils ===
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    for obj in root.findall("object"):
        difficult = int(obj.find("difficult").text)
        if difficult == 1:
            continue
        cls = obj.find("name").text.lower().strip()
        if cls not in class_to_id:
            continue
        xml_box = obj.find("bndbox")
        xmin = float(xml_box.find("xmin").text) / width
        ymin = float(xml_box.find("ymin").text) / height
        xmax = float(xml_box.find("xmax").text) / width
        ymax = float(xml_box.find("ymax").text) / height
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_id[cls])
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)

def load_image_and_targets(image_id):
    img_path = os.path.join(JPEGIMAGES_DIR, image_id + ".jpg")
    xml_path = os.path.join(ANNOTATIONS_DIR, image_id + ".xml")

    image_raw = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0

    boxes, labels = parse_annotation(xml_path)
    num_objects = boxes.shape[0]
    if num_objects > MAX_BOXES:
        boxes = boxes[:MAX_BOXES]
        labels = labels[:MAX_BOXES]
    else:
        pad_size = MAX_BOXES - num_objects
        boxes = np.pad(boxes, ((0, pad_size), (0, 0)), 'constant')
        labels = np.pad(labels, ((0, pad_size)), 'constant', constant_values=-1)

    return image, boxes, labels

def dataset_generator(txt_file):
    with open(txt_file, 'r') as f:
        image_ids = f.read().strip().split()
    for img_id in image_ids:
        yield load_image_and_targets(img_id)

def get_tf_dataset(txt_file):
    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(MAX_BOXES, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(MAX_BOXES,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(lambda: dataset_generator(txt_file), output_signature=output_signature)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# === Evaluation ===
def evaluate_model(model_path, test_file):
    model = tf.keras.models.load_model(model_path)
    test_ds = get_tf_dataset(test_file)

    y_true_all, y_pred_all = [], []
    aps_per_class = [[] for _ in range(len(VOC_CLASSES))]

    for images, true_boxes, true_labels in test_ds:
        pred_boxes, pred_logits = model(images, training=False)
        pred_labels = tf.argmax(pred_logits, axis=-1)

        for i in range(images.shape[0]):
            true_lab = true_labels[i].numpy()
            pred_lab = pred_labels[i].numpy()

            mask_true = true_lab >= 0
            mask_pred = pred_lab >= 0

            true_lab = true_lab[mask_true]
            pred_lab = pred_lab[mask_pred][:len(true_lab)]  # Match lengths for fairness

            y_true_all.extend(true_lab)
            y_pred_all.extend(pred_lab)

            # mAP per class
            for cls in range(NUM_CLASSES):
                y_true_cls = (true_lab == cls).astype(int)
                y_pred_cls = (pred_lab == cls).astype(int)
                if np.sum(y_true_cls) > 0:
                    ap = average_precision_score(y_true_cls, y_pred_cls)
                    aps_per_class[cls].append(ap)

    # Compute overall metrics
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    mAP = np.mean([np.mean(cls_aps) if cls_aps else 0 for cls_aps in aps_per_class])

    print(f"\n--- Evaluation Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP:       {mAP:.4f}")

# === Run Evaluation ===
if __name__ == "__main__":
    evaluate_model("ssd300_pascalvoc2.keras", os.path.join(IMAGESETS_MAIN, "val.txt"))
