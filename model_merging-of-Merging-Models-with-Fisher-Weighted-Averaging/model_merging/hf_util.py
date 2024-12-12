"""Utilities for HuggingFace."""
from typing import Tuple, Union

import tensorflow as tf
from transformers import TFBertPreTrainedModel
from transformers import TFRobertaPreTrainedModel


# def get_body_and_head(
#     model: Union[TFBertPreTrainedModel, TFRobertaPreTrainedModel]
# ) -> Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
#     body, *head = model.layers
#     if not head:
#         head = None
#     elif len(head) > 1:
#         raise ValueError(
#             f"Expected model to have a single 'head' layer. Instead found {len(head)}. TODO: Support this."
#         )
#     else:
#         head = head[0]
#     return body, head

# def get_body_and_head(model):
#     """
#     Extracts the body (backbone) and head (task-specific layer) of the model.
#     """
#     
#     dummy_input = tf.random.uniform(
#         [1, model.config.max_position_embeddings], dtype=tf.int32, minval=0, maxval=model.config.vocab_size
#     )
#     _ = model(dummy_input)  

#     
#     if hasattr(model, 'bert'):
#         body = model.bert
#     elif hasattr(model, 'roberta'):
#         body = model.roberta
#     else:
#         raise ValueError("Model does not have 'bert' or 'roberta' attribute.")

#     if hasattr(model, 'classifier'):
#         head = model.classifier
#     elif hasattr(model, 'classifier'):
#         head = model.classifier
#     else:
#         raise ValueError("Model does not have 'classifier' attribute.")

#     # 初始化 
#     body_outputs = body(dummy_input)
#     
#     
#     if isinstance(body_outputs, (tuple, list)):
#         
#         body_output = body_outputs[0]
#     elif isinstance(body_outputs, tf.Tensor):
#         body_output = body_outputs
#     elif hasattr(body_outputs, 'pooler_output'):
#         body_output = body_outputs.pooler_output
#     elif hasattr(body_outputs, 'last_hidden_state'):
#         body_output = body_outputs.last_hidden_state
#     else:
#         raise ValueError("Cannot find suitable output from body to pass to head.")

#     # 初始化 head 的权重
#     head(body_output)

#     return body, head

#roberta
def get_body_and_head(model):
    """
    Extracts the body (backbone) and head (task-specific layer) of the model.
    """
    #  512 for BERT and RoBERTa)
    max_input_length = min(512, model.config.max_position_embeddings - 2)  

    # Generate dummy 
    dummy_input = tf.random.uniform(
        [1, max_input_length],
        dtype=tf.int32,
        minval=0,
        maxval=model.config.vocab_size,  # maxval is exclusive
    )

    
    _ = model(dummy_input)

    
    if hasattr(model, 'roberta'):
        body = model.roberta
        
        use_pooler_output = False
    elif hasattr(model, 'bert'):
        body = model.bert
      
        use_pooler_output = True
    else:
        raise ValueError("Model does not have 'roberta' or 'bert' attribute.")

    if hasattr(model, 'classifier'):
        head = model.classifier
    else:
        raise ValueError("Model does not have 'classifier' attribute.")

    body_outputs = body(dummy_input)

    if use_pooler_output:

        body_output = body_outputs.pooler_output 
    else:

        body_output = body_outputs.last_hidden_state  

    head(body_output)

    return body, head

def get_mergeable_variables(model):
    body, _ = get_body_and_head(model)
    return body.trainable_variables


# def get_mergeable_variables(model):
#     return get_body_and_head(model)[0].trainable_variables


def clone_model(model):
    cloned = model.__class__(model.config)
    cloned(model.dummy_inputs)
    cloned.set_weights(model.get_weights())
    return cloned
