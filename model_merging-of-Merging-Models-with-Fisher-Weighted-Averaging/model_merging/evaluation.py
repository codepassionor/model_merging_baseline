"""Scripts for evaluation of models."""
import datasets as hfds
import tensorflow as tf
from evaluate import load

def load_metric_for_glue_task(task):
    # 定义任务名称的映射关系
    task_map = {
        "sst-2": "sst2",
        "mnli-mm": "mnli_mismatched",
        "mnli": "mnli_matched",
        # 根据需要添加其他映射
    }
    # 获取映射后的任务名称，如果没有映射则使用原任务名称
    task_mapped = task_map.get(task, task)
    try:
        metric = load("glue", task_mapped)
    except KeyError as e:
        raise KeyError(f"Invalid GLUE task name: {task}. Available tasks are: {list(task_map.values())}") from e
    return metric
def evaluate_model(model, dataset: tf.data.Dataset, metric):
    for model_input, gold_references in dataset:
        model_predictions = model(model_input).logits
        model_predictions = tf.argmax(model_predictions, axis=-1)
        metric.add_batch(predictions=model_predictions, references=gold_references)
    return metric.compute()
    


def average_score(score):
    return sum(score.values()) / len(score.values())
