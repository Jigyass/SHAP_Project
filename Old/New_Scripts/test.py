import tensorflow as tf

def inspect(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

inspect('/home/darksst/Desktop/SHAP_Analysis/train/data1.tfrecords')
