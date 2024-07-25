from .utils import DefaultModelVocabResizer
from .model_structure import ModelStructure

class DistilBertVocabResizer(DefaultModelVocabResizer):
    model_name : str = 'distilbert'

class DistilBertStructure(ModelStructure):
    MODEL_PREFIX: str = "distilbert."
    ENCODER_PREFIX: str = r"transformer.layer.[0-9]+\."
    LAYER_PATTERNS = dict(
        query="attention.q_lin",
        key="attention.k_lin",
        value="attention.v_lin",
        att_dense="attention.out_lin",
        interm_dense="ffn.lin1",
        output_dense="ffn.lin2",
    )
    ATTENTION_PREFIX = ("attention",)
    ATTENTION_LAYERS = ("q_lin", "k_lin", "v_lin")
    MHA_LAYERS = ATTENTION_LAYERS #+ ("attention",)
    NAME_CONFIG = dict(
        hidden_size="hidden_size",
        intermediate_size="hidden_size",
        num_hidden_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        attention_head_size="attention_head_size",
    )