import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
class_to_id = {cls: i for i, cls in enumerate(VOC_CLASSES)}

IMG_SIZE = 300
NUM_CLASSES = len(VOC_CLASSES)
BATCH_SIZE = 8
MAX_BOXES = 10

VOC_ROOT = r"C:\Users\dubey\Downloads\archive\VOC2012"
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
JPEGIMAGES_DIR = os.path.join(VOC_ROOT, "JPEGImages")
IMAGESETS_MAIN = os.path.join(VOC_ROOT, "ImageSets", "Main")

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes, labels = [], []

    size = root.find("size")
    width, height = float(size.find("width").text), float(size.find("height").text)

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
    image = tf.image.resize(image, [int(IMG_SIZE), int(IMG_SIZE)])
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
    ds = ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_ssd300():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    x = tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(base_model.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    class_out = tf.keras.layers.Dense(MAX_BOXES * NUM_CLASSES, activation="softmax")(x)
    class_out = tf.keras.layers.Reshape((MAX_BOXES, NUM_CLASSES))(class_out)

    box_out = tf.keras.layers.Dense(MAX_BOXES * 4, activation="sigmoid")(x)
    box_out = tf.keras.layers.Reshape((MAX_BOXES, 4))(box_out)

    return tf.keras.Model(inputs=base_model.input, outputs=[box_out, class_out])

def ssd_loss(true_boxes, pred_boxes, true_labels, pred_logits):
    mask = tf.cast(true_labels >= 0, tf.float32)
    box_loss = tf.reduce_sum(tf.abs(true_boxes - pred_boxes), axis=-1)
    box_loss = tf.reduce_sum(box_loss * mask) / (tf.reduce_sum(mask) + 1e-6)

    true_labels_clipped = tf.maximum(true_labels, 0)
    true_labels_onehot = tf.one_hot(true_labels_clipped, NUM_CLASSES)
    class_loss = tf.keras.losses.categorical_crossentropy(
        true_labels_onehot, pred_logits, from_logits=False, axis=-1
    )
    class_loss = tf.reduce_sum(class_loss * mask) / (tf.reduce_sum(mask) + 1e-6)
    return box_loss + class_loss

def train(model, train_ds, val_ds, epochs=10):
    optimizer = tf.keras.optimizers.Adam(1e-4)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch, (images, boxes, labels) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                pred_boxes, pred_logits = model(images, training=True)
                loss = ssd_loss(boxes, pred_boxes, labels, pred_logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if batch % 10 == 0:
                print(f"Batch {batch} Loss: {loss.numpy():.4f}")

        # Validation
        total_correct, total_objects = 0, 0
        for images, boxes, labels in val_ds:
            pred_boxes, pred_logits = model(images, training=False)
            pred_labels = tf.argmax(pred_logits, axis=-1)
            mask = tf.cast(labels >= 0, tf.bool)
            correct = tf.reduce_sum(tf.cast(tf.equal(labels, tf.cast(pred_labels, tf.int32)) & mask, tf.int32)).numpy()
            total_correct += correct
            total_objects += tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()
        val_acc = total_correct / (total_objects + 1e-6)
        print(f"Validation Accuracy: {val_acc:.4f}")



def test(model, test_ds):
    print("Running test evaluation...")
    total_correct, total_objects = 0, 0
    for images, boxes, labels in test_ds:
        pred_boxes, pred_logits = model(images, training=False)
        pred_labels = tf.argmax(pred_logits, axis=-1)
        mask = tf.cast(labels >= 0, tf.bool)
        correct = tf.reduce_sum(tf.cast(tf.equal(labels, tf.cast(pred_labels, tf.int32)) & mask, tf.int32)).numpy()
        total_correct += correct
        total_objects += tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()
    test_acc = total_correct / (total_objects + 1e-6)
    print(f"Test Accuracy: {test_acc:.4f}")



if __name__ == "__main__":
    train_ds = get_tf_dataset(os.path.join(IMAGESETS_MAIN, "train.txt"))
    val_ds = get_tf_dataset(os.path.join(IMAGESETS_MAIN, "val.txt"))
    test_ds = get_tf_dataset(os.path.join(IMAGESETS_MAIN, "trainval.txt"))

    model = build_ssd300()
    train(model, train_ds, val_ds, epochs=5)
    test(model, test_ds)
    model.save("ssd300_pascalvoc2.keras")
    print("Model saved as ssd300_pascalvoc2.keras")
