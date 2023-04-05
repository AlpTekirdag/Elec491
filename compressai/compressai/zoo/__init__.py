from .image import (
    bmshj2018_saliency,
    bmshj2018_factorized,
    bmshj2018_factorized_relu,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean,
)
from .pretrained import load_pretrained as load_state_dict
from .video import ssf2020

image_models = {
    "bmshj2018-saliency": bmshj2018_saliency,
    "bmshj2018-saliency-modulate": bmshj2018_saliency_modulate,
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-factorized-relu": bmshj2018_factorized_relu,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
}

video_models = {
    "ssf2020": ssf2020,
}

models = {}
models.update(image_models)
models.update(video_models)
