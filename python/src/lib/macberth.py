"""
lib/macberth.py
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union
import onnxruntime as ort
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from lib.corpus_logging import logger
from lib.corpus_config import MODELS_DIR

MACBERTH_MODEL_PATH = Path("./lib/macberth-huggingface")
MACBERTH_MODEL_NAME = "emanjavacas/MacBERTh"

ONNX_MODEL_DIR = MODELS_DIR / "./macberth-onnx-fp32"
ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

ONNX_INT8_MODEL_DIR = MODELS_DIR / "./macberth-onnx-int8"
ONNX_INT8_MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64

@dataclass
class MacberthModel:
    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM
    device: str

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def encode(self, **kwargs):
        """
        Run the encoder only.

        Used by Tier 1:
            - normal tokens
            - masked tokens

        Returns hidden states.
        """
        return self.model.base_model(**kwargs)

    def predict_masked(self, **kwargs):
        """
        Run the full masked-language model.

        Used by Tier 1.4 substitute vectors.

        Returns logits over vocabulary.
        """
        return self.model(**kwargs)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_macberth(
    *,
    use_qint8: bool = False,
) -> MacberthModel:
    """
    Loads MacBERTh with encoder + MLM head.

    The encoder is accessed through model.base_model.

    (The MLM head is currently used by Tier 1.4.)
    """

    logger.info("Loading MacBERTh model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Tokenizer must be fast")

    model = AutoModelForMaskedLM.from_pretrained(
        MACBERTH_MODEL_PATH,
        local_files_only=True,
    )

    if use_qint8:
        raise RuntimeError("Do not use q8 it is too noisy to have any value")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("Set model to use qint8")

    device = get_device()

    model.to(device)
    model.eval()

    return MacberthModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


class MacBERThEmbedder:
    """
    BERTopic-compatible wrapper around MacberthModel.
    """

    def __init__(self, macberth: MacberthModel, pooling: str = "mean"):
        self.macberth = macberth
        self.pooling = pooling
        self.device = macberth.device


    @property
    def hidden_size(self) -> int:
        return self.macberth.hidden_size

    # def encode(
    #     self,
    #     texts: Union[str, List[str]],
    #     show_progress_bar: bool = False,
    #     convert_to_numpy: bool = True,
    #     **kwargs,
    # ) -> np.ndarray:

    #     if isinstance(texts, str):
    #         texts = [texts]

    #     all_embeddings = []

    #     with torch.no_grad():
    #         for text in texts:

    #             encoded = self.macberth.tokenizer(
    #                 text,
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=512,
    #                 return_tensors="pt",
    #             )

    #             encoded = {
    #                 k: v.to(self.device)
    #                 for k, v in encoded.items()
    #             }

    #             # Encoder only, not MLM head
    #             outputs = self.macberth.encode(**encoded)

    #             if self.pooling == "cls":
    #                 emb = outputs.last_hidden_state[:, 0, :]

    #             elif self.pooling == "max":
    #                 emb = outputs.last_hidden_state.max(dim=1).values

    #             else:
    #                 attention_mask = encoded["attention_mask"]
    #                 emb = self._mean_pooling(
    #                     outputs.last_hidden_state,
    #                     attention_mask,
    #                 )

    #             all_embeddings.append(
    #                 emb.cpu().numpy().squeeze()
    #             )

    #     embeddings = np.array(all_embeddings)

    #     if convert_to_numpy:
    #         return embeddings

    #     return torch.tensor(embeddings)


    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[start:start + BATCH_SIZE]

                encoded = self.macberth.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                encoded = {
                    k: v.to(self.device)
                    for k, v in encoded.items()
                }

                outputs = self.macberth.encode(**encoded)

                if self.pooling == "cls":
                    emb = outputs.last_hidden_state[:, 0, :]

                elif self.pooling == "max":
                    emb = outputs.last_hidden_state.max(dim=1).values

                else:
                    attention_mask = encoded["attention_mask"]
                    emb = self._mean_pooling(
                        outputs.last_hidden_state,
                        attention_mask,
                    )

                all_embeddings.extend(emb.cpu().numpy())

        embeddings = np.array(all_embeddings)

        if convert_to_numpy:
            return embeddings

        return torch.tensor(embeddings)



    def encode_normalized(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:

        embeddings = self.encode(texts)
        return np.array([
            normalize(v)
            for v in embeddings
        ])


    @staticmethod
    def _mean_pooling(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(hidden_states.size())
            .float()
        )

        sum_embeddings = torch.sum(
            hidden_states * input_mask_expanded,
            dim=1,
        )

        sum_mask = torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

        return sum_embeddings / sum_mask


class OnnxMacberthModel:
    """
    ONNX Runtime-backed stand-in for MacberthModel. Exposes the same
    .tokenizer / .device / .encode() surface so MacBERThEmbedder doesn't
    need to know the difference.
    """

    def __init__(self, tokenizer, session):
        self.tokenizer = tokenizer
        self.session = session

        self.device = "cpu"

        # Read once from ONNX metadata rather than requiring transformers config.
        # The exported encoder output shape is [batch, sequence, hidden_size].
        output_shape = session.get_outputs()[0].shape
        self._hidden_size = output_shape[-1]

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def encode(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        # This model is CPU-only (self.device == "cpu"), so input_ids /
        # attention_mask / token_type_ids never need to round-trip through
        # torch here -- go straight to numpy, which is all ORT accepts
        # anyway. This runs once per batch on the hot path.
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()

        if token_type_ids is None:
            token_type_ids_np = np.zeros_like(input_ids_np)
        else:
            token_type_ids_np = token_type_ids.cpu().numpy()

        feed = {
            "input_ids": input_ids_np,
            "attention_mask": attention_mask_np,
            "token_type_ids": token_type_ids_np,
        }

        outputs = self.session.run(None, feed)

        last_hidden_state = torch.from_numpy(outputs[0])

        return SimpleNamespace(
            last_hidden_state=last_hidden_state
        )


def _export_macberth_onnx(export_dir: Path = ONNX_MODEL_DIR) -> None:
    """
    Exports a fresh fp32 (unquantized) MacBERTh to ONNX format.

    Loads its own fp32 model instance rather than dequantizing an
    existing one -- dynamic quantization (use_qint8=True) repacks
    Linear weights into a non-tensor format with no clean reverse op.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    import shutil

    logger.info(f"ONNX export not found at {export_dir}, exporting now...")

    if export_dir.exists():
        shutil.rmtree(export_dir)

    mac_fp32 = load_macberth(use_qint8=False)
    mac_fp32.model.eval()

    temp_dir = Path(str(export_dir) + "_temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    mac_fp32.model.save_pretrained(temp_dir)

    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        temp_dir,
        export=True,
        provider="CPUExecutionProvider",
    )
    ort_model.save_pretrained(export_dir)
    mac_fp32.tokenizer.save_pretrained(export_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(f"ONNX export completed at {export_dir}")


def _quantize_macberth_onnx(
    fp32_dir: Path = ONNX_MODEL_DIR,
    int8_dir: Path = ONNX_INT8_MODEL_DIR,
) -> None:
    """
    Dynamically quantizes the fp32 ONNX export to int8 (weights only;
    activations are quantized on the fly at inference time).

    Dynamic quantization terrible - do not use it.

    $ python src/tests/test_quantization_noise_vs_drift.py
    [quant-test] Loading fp32 and int8 ONNX models...
    [macberth] ORT session config: intra_op_num_threads=8, inter_op_num_threads=1, execution_mode=SEQUENTIAL, spinning=off
    [macberth.load_macberth_onnx] Loaded ONNX MacBERTh (quantize=False), providers: ['CPUExecutionProvider']
    [macberth] ORT session config: intra_op_num_threads=8, inter_op_num_threads=1, execution_mode=SEQUENTIAL, spinning=off
    [macberth.load_macberth_onnx] Loaded ONNX MacBERTh (quantize=True), providers: ['CPUExecutionProvider']

    Embedding 6 sentence(s) with both models...

    [0.45867] The king did graciously receive the petition of his loyal subjects.
    [0.46124] It is reported that the plague hath spread through the parish.
    [0.46857] Concerning the nature of witchcraft, many learned men have written.
    [0.46696] The merchant sold his wares at the market on Thursday last.
    [0.47593] Let all men know that this covenant is binding before God.
    [0.50061] The soldiers marched upon the town at break of day.

    === Summary (as cosine distance, 1 - similarity) ===

    NOISE      (fp32 vs int8, same sentence): n=6  mean=0.52800  p5=0.50556  median=0.53223  p95=0.54068
    REFERENCE  (fp32 vs fp32, different sentences): n=15  mean=0.11196  p5=0.08632  median=0.10968  p95=0.14221

    noise_median / reference_median = 4.8527

    Note: 'reference' here is the gap between semantically different sentences, which is a coarse scale check, not a measurement of real diachronic drift. A small ratio is reassuring but doesn't by itself confirm int8 is safe for subtle across-year semantic-change analysis -- that specifically needs real corpus occurrences across years.
$

    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import shutil

    if not (fp32_dir / "model.onnx").exists():
        _export_macberth_onnx(fp32_dir)

    logger.info(f"Quantizing ONNX MacBERTh to int8 at {int8_dir}...")

    if int8_dir.exists():
        shutil.rmtree(int8_dir)
    int8_dir.mkdir(parents=True, exist_ok=True)

    # ORT recommends running shape-inference/node-fusion pre-processing
    # before dynamic quantization; skipping it (the previous version of
    # this function did) measurably degrades int8 accuracy. This writes
    # an intermediate "preprocessed" fp32 model that quantize_dynamic then
    # reads from, rather than quantizing the raw export directly.
    #
    # Symbolic shape inference (skip_symbolic_shape=False) is more
    # thorough but commonly fails on transformer exports that contain
    # dynamic reshapes/attention ops it can't fully resolve -- this is a
    # known ORT limitation, not specific to this model. Fall back to
    # basic (non-symbolic) shape inference in that case rather than
    # aborting quantization entirely.
    preprocessed_path = int8_dir / "model.preprocessed.onnx"

    try:
        quant_pre_process(
            input_model=str(fp32_dir / "model.onnx"),
            output_model_path=str(preprocessed_path),
            skip_symbolic_shape=False,
        )
    except Exception as e:
        logger.warning(
            "[macberth] Symbolic shape inference failed during "
            "quant_pre_process (%s); falling back to basic "
            "(non-symbolic) shape inference. This is a known ORT "
            "limitation on some transformer exports.",
            e,
        )
        quant_pre_process(
            input_model=str(fp32_dir / "model.onnx"),
            output_model_path=str(preprocessed_path),
            skip_symbolic_shape=True,
        )

    # NOTE: per_channel=True was tried here and rejected. On this model's
    # exported graph it produced badly broken embeddings (same-sentence
    # fp32-vs-int8 cosine similarity ~0.46, i.e. near-uncorrelated) rather
    # than a subtler accuracy change. This matches a known ORT failure
    # mode where per-channel dynamic quantization assumes a particular
    # weight layout and misapplies scales along the wrong axis on
    # transposed-MatMul graphs (common in HF/optimum ONNX exports that
    # use MatMul rather than Gemm). Do not re-enable without first
    # fixing the graph layout (e.g. via a Gemm-producing export path or
    # explicit axis correction) -- otherwise it silently corrupts output
    # rather than just losing accuracy.
    quantize_dynamic(
        model_input=str(preprocessed_path),
        model_output=str(int8_dir / "model.onnx"),
        weight_type=QuantType.QInt8,
    )

    preprocessed_path.unlink(missing_ok=True)

    # Tokenizer / config files live alongside the model but aren't part
    # of the quantization step -- copy them over so int8_dir is a
    # complete, independently loadable export.
    for pattern in ("*.json", "*.txt"):
        for f in fp32_dir.glob(pattern):
            shutil.copy(f, int8_dir / f.name)

    logger.info(f"ONNX int8 quantization completed at {int8_dir}")


def _configure_ort_session_options() -> ort.SessionOptions:
    """
    Centralizes CPU thread/execution tuning for the ONNX session.

    Rationale (all CPU-bound-specific):
      - intra_op_num_threads is set explicitly rather than left to ORT's
        default, and is coordinated with cpu_count() so it doesn't
        silently pick a value that fights the rest of the pipeline
        (tokenization, window bookkeeping, parquet writes) for cores.
        ORT_NUM_THREADS env var still overrides if set.
      - inter_op_num_threads=1 because a single BERT encoder graph has
        essentially no independent branches to parallelize across --
        inter-op parallelism here is pure scheduling overhead.
      - ORT_SEQUENTIAL over the default ORT_PARALLEL for the same reason:
        parallel execution mode is built for graphs with concurrent
        branches, not a single linear encoder stack.
      - allow_spinning=0 so idle ORT worker threads yield the CPU
        instead of busy-waiting, which matters because this process
        interleaves non-trivial Python work between forward passes
        rather than running back-to-back inference in a tight loop.
    """
    import os

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        n = int(os.environ.get("ORT_NUM_THREADS", "0"))
    except ValueError:
        n = 0

    if n <= 0:
        n = os.cpu_count() or 4

    sess_options.intra_op_num_threads = n
    sess_options.inter_op_num_threads = 1

    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    logger.info(
        "[macberth] ORT session config: intra_op_num_threads=%d, "
        "inter_op_num_threads=1, execution_mode=SEQUENTIAL, spinning=off",
        n,
    )

    return sess_options


def load_macberth_onnx(
    export_dir: Optional[Path] = None,
    providers: Optional[List[str]] = None,
    *,
    quantize: bool = False,
) -> OnnxMacberthModel:
    """
    Loads the ONNX MacBERTh model for inference. Exports it first if it
    doesn't already exist on disk.

    quantize=True (default) uses the dynamically-quantized int8 export,
    which is substantially faster on CPU for an encoder-only transformer
    like this one -- exactly the case here, since DirectML is unusable
    and CPU is the only viable backend. Pass quantize=False to fall back
    to the fp32 export (e.g. if you need to A/B against int8 output).

    Default provider is CPU-only (stable for long Tier 1 runs on Windows).
    Pass `providers=["DmlExecutionProvider", "CPUExecutionProvider"]` or set
    `MACBERTH_ONNX_PROVIDER=dml` to use DirectML.
    """
    import os

    if export_dir is None:
        export_dir = ONNX_INT8_MODEL_DIR if quantize else ONNX_MODEL_DIR
    export_dir = Path(export_dir)

    if not (export_dir / "model.onnx").exists():
        if quantize:
            _quantize_macberth_onnx(ONNX_MODEL_DIR, export_dir)
        else:
            _export_macberth_onnx(export_dir)

    if providers is None:
        # Env override: MACBERTH_ONNX_PROVIDER=dml|cpu
        pref = os.environ.get("MACBERTH_ONNX_PROVIDER", "cpu").strip().lower()
        if pref in ("dml", "directml", "gpu"):
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    usable = [p for p in providers if p in ort.get_available_providers()]
    if not usable:
        raise RuntimeError(f"None of the requested providers are available: {providers}")

    provider_options = [
        {"device_id": 0} if p == "DmlExecutionProvider" else {}
        for p in usable
    ]

    sess_options = _configure_ort_session_options()

    session = ort.InferenceSession(
        f"{export_dir}/model.onnx",
        sess_options=sess_options,
        providers=usable,
        provider_options=provider_options,
    )
    logger.info(
        "[macberth.load_macberth_onnx] Loaded ONNX MacBERTh (quantize=%s), providers: %s",
        quantize,
        session.get_providers(),
    )

    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)

    return OnnxMacberthModel(tokenizer=tokenizer, session=session)


def get_macberth_embedder(
    pooling: str = "mean",
    backend: str = "onnx",  # "onnx" or "pytorch"
) -> MacBERThEmbedder:

    if backend == "onnx":
        macberth_model = load_macberth_onnx()
    else:
        macberth_model = load_macberth()

    return MacBERThEmbedder(
        macberth_model,
        pooling=pooling,
    )


def embed_query(
    text: str,
    *,
    backend: str = "onnx",
    pooling: str = "mean",
) -> np.ndarray:
    """
    Encode a short natural-language query for vector retrieval.

    Returns a single L2-normalized vector with shape (1, hidden_size),
    suitable for FAISS inner-product search.

    Query vectors must follow the same normalization convention as the
    stored Tier 1 vectors; otherwise inner-product scores are not comparable.
    """

    embedder = get_macberth_embedder(
        pooling=pooling,
        backend=backend
    )

    embedding = embedder.encode( text )[0]
    vector = normalize(embedding)

    if vector is None:
        raise RuntimeError( "MacBERTh produced zero-length query embedding" )

    return vector.astype(np.float32)[None, :]
