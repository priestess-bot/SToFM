from typing import Optional

from transformers.configuration_utils import PretrainedConfig
import torch

try:
    from geneformer import DataCollatorForCellClassification
    from geneformer.collator_for_classification import PrecollatorForGeneAndCellClassification
    from geneformer.pretrainer import token_dictionary
except ImportError as exc:
    _GENEFORMER_IMPORT_ERROR = exc
else:
    _GENEFORMER_IMPORT_ERROR = None
    token_dictionary["<cls>"] = len(token_dictionary)


def _require_geneformer() -> None:
    if _GENEFORMER_IMPORT_ERROR is not None:
        raise ImportError(
            "CellEncoderTokenizer and CellEncoderCollator require the optional Geneformer dependencies"
        ) from _GENEFORMER_IMPORT_ERROR


if _GENEFORMER_IMPORT_ERROR is None:

    class CellEncoderTokenizer(PrecollatorForGeneAndCellClassification):
        cls_token = "<cls>"
        cls_token_id = token_dictionary.get("<cls>")
        all_special_ids = [
            token_dictionary.get("<cls>"),
            token_dictionary.get("<mask>"),
            token_dictionary.get("<pad>"),
        ]
        token_dictionary = token_dictionary

        def _convert_token_to_id_with_added_voc(self, token):
            if token is None:
                return None
            return self.token_dictionary.get(token)

        def __len__(self):
            return len(self.token_dictionary)


    class CellEncoderCollator(DataCollatorForCellClassification):
        def __init__(self, add_cls=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_cls = add_cls
            self.tokenizer = CellEncoderTokenizer()

        def _prepare_batch(self, features):
            if self.add_cls:
                for feature in features:
                    feature["input_ids"] = (
                        [int(self.tokenizer.cls_token_id)] + feature["input_ids"]
                    )[:2048]

            batch = super()._prepare_batch(features)
            first = features[0]
            if "label" in first and first["label"] is not None:
                if isinstance(first["label"], torch.Tensor):
                    label = first["label"].item()
                elif isinstance(first["label"], list):
                    label = first["label"][0]
                else:
                    label = first["label"]
                dtype = torch.long if isinstance(label, int) else torch.float
                batch["labels"] = torch.tensor(
                    [feature["label"] for feature in features], dtype=dtype
                )
            return batch

else:

    class CellEncoderTokenizer:
        def __init__(self, *args, **kwargs):
            _require_geneformer()


    class CellEncoderCollator:
        def __init__(self, *args, **kwargs):
            _require_geneformer()

class SToFMConfig(PretrainedConfig):

    def __init__(
        self,
        num_hidden_layers: int = 12,
        input_dim: int = 256,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        gaussian_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        num_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        flagos_backend: str = "torch",
        flagos_attention_backend: Optional[str] = None,
        norm_type_id=0,
        cls_type_id=1,
        hyper_type_id=2,
        pad_type_id=3,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.gaussian_hidden_dim = gaussian_hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_init = apply_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias
        self.flagos_backend = flagos_backend
        self.flagos_attention_backend = (
            flagos_backend if flagos_attention_backend is None else flagos_attention_backend
        )

        self.norm_type_id = norm_type_id
        self.cls_type_id = cls_type_id
        self.hyper_type_id = hyper_type_id
        self.pad_type_id = pad_type_id

        super().__init__(
            **kwargs,
        )
