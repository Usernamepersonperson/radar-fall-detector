"""Fall classifier: small MLP on window features, exported to ONNX (hand-built graph).

A gradient-boosting model is trained as a reference baseline only (not deployed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from .features import N_FEAT

MODELS = Path(__file__).resolve().parent.parent / "models"
ONNX_PATH = MODELS / "fall_mlp.onnx"
META_PATH = MODELS / "meta.json"


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int = 0, pos_rep: int = 6):
    """Fit standardiser + MLP(32,16). Positives are replicated ``pos_rep`` times."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    idx = np.concatenate([np.arange(len(y)), np.repeat(np.where(y == 1)[0],
                                                       pos_rep - 1)])
    clf = MLPClassifier((32, 16), alpha=1e-3, max_iter=400, random_state=seed,
                        early_stopping=True, n_iter_no_change=20)
    clf.fit(xs[idx], y[idx])
    return clf, mu.astype(np.float32), sd.astype(np.float32)


def train_gb(x: np.ndarray, y: np.ndarray, seed: int = 0):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                         class_weight="balanced", random_state=seed)
    clf.fit(x, y)
    return clf


def to_onnx(clf: MLPClassifier, mu: np.ndarray, sd: np.ndarray) -> onnx.ModelProto:
    """standardise -> (Gemm, Relu)* -> Gemm -> Sigmoid. Input 'x' (N, F) float32."""
    inits = [numpy_helper.from_array(mu, "mu"), numpy_helper.from_array(sd, "sd")]
    nodes = [helper.make_node("Sub", ["x", "mu"], ["xc"]),
             helper.make_node("Div", ["xc", "sd"], ["h0"])]
    h = "h0"
    n_layers = len(clf.coefs_)
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        inits += [numpy_helper.from_array(w.astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        out = f"z{i}"
        nodes.append(helper.make_node("Gemm", [h, f"W{i}", f"b{i}"], [out]))
        if i < n_layers - 1:
            nodes.append(helper.make_node("Relu", [out], [f"h{i + 1}"]))
            h = f"h{i + 1}"
        else:
            nodes.append(helper.make_node("Sigmoid", [out], ["p_fall"]))
    graph = helper.make_graph(
        nodes, "radar_fall_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, N_FEAT])],
        [helper.make_tensor_value_info("p_fall", TensorProto.FLOAT, [None, 1])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def paths_for(scale_name: str) -> tuple[Path, Path]:
    """Committed model for the full scale; quick/smoke runs stay out of models/."""
    if scale_name == "full":
        return ONNX_PATH, META_PATH
    d = MODELS.parent / "data" / scale_name
    return d / "fall_mlp.onnx", d / "meta.json"


def save_model(m: onnx.ModelProto, meta: dict, path: Path = ONNX_PATH,
               meta_path: Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))
    (meta_path or path.with_name("meta.json")).write_text(json.dumps(meta, indent=2))


class OnnxScorer:
    """onnxruntime wrapper: features -> P(fall)."""

    def __init__(self, path: Path | str = ONNX_PATH):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so,
                                         providers=["CPUExecutionProvider"])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return self.sess.run(None, {"x": x})[0][:, 0]
