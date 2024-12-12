# model_merging/data.py
"""Code for loading data, focusing on GLUE."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset

class Sst2Processor:
    """Processor for the SST-2 data set (GLUE version)."""

    def get_examples(self, data_dir=None, split='train'):
        """Load examples from the SST-2 dataset."""
        dataset = load_dataset("glue", "sst2", split=split)
        return dataset

    def get_labels(self):
        """Get the list of labels for SST-2."""
        return ["negative", "positive"]

    def get_example_from_tensor_dict(self, tensor_dict):
        """Converts a TensorFlow tensor dictionary to a simple dict."""
        sentence = tensor_dict["sentence"].numpy().decode('utf-8')
        label = tensor_dict["label"].numpy()
        return {
            "sentence": sentence,
            "label": label,
        }

    def tfds_map(self, example):
        """Further processes the example if needed."""
        return example

from transformers.data.processors.glue import (
    ColaProcessor,
    MnliProcessor,
    MrpcProcessor,
    QnliProcessor,
    QqpProcessor,
    RteProcessor,
    StsbProcessor,
    WnliProcessor,
)
# tasks reference
_glue_processors = {
    'cola': ColaProcessor,
    'mnli': MnliProcessor,
    'mrpc': MrpcProcessor,
    'qnli': QnliProcessor,
    'qqp': QqpProcessor,
    'rte': RteProcessor,
    'sst-2': Sst2Processor,
    'sts-b': StsbProcessor,
    'wnli': WnliProcessor,
}

_glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli_matched": "classification",
    "mnli_mismatched": "classification",
    "mrpc": "classification",
    "qnli": "classification",
    "qqp": "classification",
    "rte": "classification",
    "sst-2": "classification",
    "stsb": "regression",
    "wnli": "classification",
    "hans": "classification",
}

_STSB_MIN = 0
_STSB_MAX = 5
_STSB_NUM_BINS = 5 * (_STSB_MAX - _STSB_MIN)

def _to_tfds_task_name(task, split):
    if task == "sts-b":
        task = "stsb"
    elif task == "sst-2":
        task = "sst2"
    elif task == "mnli" and split != "train":
        task = "mnli_matched"
    elif task == "mnli-mm" and split != "train":
        task = "mnli_mismatched"
    return task

def _convert_dataset_to_features(
    dataset,
    tokenizer,
    max_length,
    task,
):
    print(f"Converting dataset for task: {task}")
    processor = _glue_processors[task]()
    labels = processor.get_labels()
    output_mode = _glue_output_modes[task]

    if task == "sts-b":
        # STS-B regression.
        stsb_bins = np.linspace(_STSB_MIN, _STSB_MAX, num=_STSB_NUM_BINS + 1)
        stsb_bins = stsb_bins[1:-1]
    else:
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

    def py_map_fn(keys, *values):
        example = {tf.compat.as_str(k.numpy()): v for k, v in zip(keys, values)}
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

        if task == "sst-2":
            inputs = tokenizer.encode_plus(
                example['sentence'],
                None,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
                truncation=True,
            )
        else:
            inputs = tokenizer.encode_plus(
                example['sentence1'],
                example['sentence2'],
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
                truncation=True,
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        input_ids = tf.constant(input_ids, dtype=tf.int32)
        token_type_ids = tf.constant(token_type_ids, dtype=tf.int32)

        if output_mode == "classification":
            if task == "sst-2":
                label = tf.constant(example['label'], dtype=tf.int64)
            else:
                label = label_map[example['label']]
                label = tf.constant(label, dtype=tf.int64)
        else:
            label = float(example['label'])
            assert 0.0 <= label <= 5.0, f"Out of range STS-B label {label}."
            label = np.digitize(label, stsb_bins)
            label = tf.constant(label, dtype=tf.int64)
        return input_ids, token_type_ids, label

    def map_fn(example):
        input_ids, token_type_ids, label = tf.py_function(
            func=py_map_fn,
            inp=[list(example.keys()), *example.values()],
            Tout=[tf.int32, tf.int32, tf.int64],
        )
        return input_ids, token_type_ids, label

    pad_token = tokenizer.pad_token_id
    pad_token_segment_id = tokenizer.pad_token_type_id

    def pad_fn(input_ids, token_type_ids, label):
        # Zero-pad up to the sequence length.
        padding_length = max_length - tf.shape(input_ids)[-1]

        pad_token_tf = tf.constant(pad_token, dtype=tf.int32)
        pad_token_segment_id_tf = tf.constant(pad_token_segment_id, dtype=tf.int32)

        input_ids = tf.concat(
            [input_ids, pad_token_tf * tf.ones(padding_length, dtype=tf.int32)], axis=-1
        )
        token_type_ids = tf.concat(
            [
                token_type_ids,
                pad_token_segment_id_tf * tf.ones(padding_length, dtype=tf.int32),
            ],
            axis=-1,
        )

        tf_example = {
            "input_ids": tf.reshape(input_ids, [max_length]),
            "token_type_ids": tf.reshape(token_type_ids, [max_length]),
        }
        return tf_example, label

    dataset = dataset.map(map_fn)
    dataset = dataset.map(pad_fn)
    return dataset

def load_glue_dataset(task: str, split: str, tokenizer, max_length: int):
    tfds_task = _to_tfds_task_name(task, split)
    print(f"Loading GLUE task: {task}, tfds_task: {tfds_task}")
    ds = tfds.load(f"glue/{tfds_task}", split=split)
    ds = _convert_dataset_to_features(
        ds,
        tokenizer,
        max_length,
        task,
    )
    return ds
