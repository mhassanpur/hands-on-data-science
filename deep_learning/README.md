# Deep Learning, Built From Scratch

Seventeen notebook chapters that start with a single neuron trained by
hand-derived gradients and end at the frontier: the architecture behind Stable
Diffusion 3, models that see and talk, alternatives to attention itself, and the
engineering that makes any of it fit on one GPU. Nothing is imported before it
has been built at least once.

Each chapter is a standalone notebook that redefines what it needs, but the
ideas are strictly cumulative, and each wrap-up sets up the next.

## The chapters, in order

### Part I: Supervised foundations

| # | Notebook | What you build | Dataset |
|---|---|---|---|
| 1 | [ch01_intro_neural_networks.ipynb](ch01_intro_neural_networks.ipynb) | A neuron, then an MLP with hand-written backprop; the same model in Keras; overfitting and a tabular pipeline | synthetic |
| 2 | [ch02_convolutional_neural_network.ipynb](ch02_convolutional_neural_network.ipynb) | `conv2d` in NumPy, a CNN vs a dense baseline, VGG-style blocks, filter and feature-map inspection | `ylecun/mnist`, `uoft-cs/cifar10` |
| 3 | [ch03_residual_networks.ipynb](ch03_residual_networks.ipynb) | The vanishing gradient measured; ResNet-32 vs plain-32, raced | `uoft-cs/cifar10` |
| 4 | [ch04_transfer_learning_and_finetuning.ipynb](ch04_transfer_learning_and_finetuning.ipynb) | A frozen pretrained backbone, then the two-stage fine-tuning recipe | `AI-Lab-Makerere/beans` |

### Part II: Generative models

| # | Notebook | What you build | Dataset |
|---|---|---|---|
| 5 | [ch05_autoencoders_and_latent_spaces.ipynb](ch05_autoencoders_and_latent_spaces.ipynb) | PCA baseline, conv autoencoder, latent cartography, and the sampling **failure** | `zalando-datasets/fashion_mnist` |
| 6 | [ch06_variational_autoencoders.ipynb](ch06_variational_autoencoders.ipynb) | The reparameterization trick, the KL term, a custom `train_step`, a denoising AE | `zalando-datasets/fashion_mnist` |
| 7 | [ch07_generative_adversarial_networks.ipynb](ch07_generative_adversarial_networks.ipynb) | The adversarial game on a 2-D ring, a DCGAN, mode collapse measured | `zalando-datasets/fashion_mnist` |
| 8 | [ch08_diffusion_models.ipynb](ch08_diffusion_models.ipynb) | Forward diffusion, a U-Net denoiser, DDPM sampling, DDIM, classifier-free guidance | `ylecun/mnist` |

### Part III: Attention

| # | Notebook | What you build | Dataset |
|---|---|---|---|
| 9 | [ch09_self_attention.ipynb](ch09_self_attention.ipynb) | $Q,K,V$ by hand; scaled dot-product attention; masking; the $O(n^2)$ bill | `stanfordnlp/imdb` |
| 10 | [ch10_multi_head_attention.ipynb](ch10_multi_head_attention.ipynb) | Positional encodings, multi-head attention, the full encoder block | `stanfordnlp/imdb` |
| 11 | [ch11_vision_transformers.ipynb](ch11_vision_transformers.ipynb) | Patch embedding, `[CLS]`, ViT, and the inductive-bias experiment | `uoft-cs/cifar10` |
| 12 | [ch12_diffusion_transformers.ipynb](ch12_diffusion_transformers.ipynb) | adaLN-Zero, the DiT block, training, sampling, guidance | `zalando-datasets/fashion_mnist` |

### Part IV: Multimodal, alternatives, and engineering

| # | Notebook | What you build | Dataset |
|---|---|---|---|
| 13 | [ch13_clip_contrastive_pretraining.ipynb](ch13_clip_contrastive_pretraining.ipynb) | Two towers into one space, the InfoNCE loss, zero-shot classification, the **modality gap** | `zalando-datasets/fashion_mnist` |
| 14 | [ch14_vision_language_models.ipynb](ch14_vision_language_models.ipynb) | LLaVA in miniature: frozen towers, a projector, generated captions, the **blind baseline** | `zalando-datasets/fashion_mnist` |
| 15 | [ch15_state_space_models_mamba.ipynb](ch15_state_space_models_mamba.ipynb) | The parallel scan, discretization, Mamba's selection, and recall vs attention | synthetic |
| 16 | [ch16_lora_and_qlora.ipynb](ch16_lora_and_qlora.ipynb) | LoRA from scratch, rank sweeps, merging, blockwise NF4 quantization, QLoRA | `fashion_mnist` → `ylecun/mnist` |
| 17 | [ch17_flash_attention.ipynb](ch17_flash_attention.ipynb) | Online softmax, tiled attention verified against torch's kernel, and the IO model | — |

**The shape of the argument.** Chapters 1–4 are supervised: input in, label out.
Chapters 5–8 are three different answers to one problem that Chapter 5 states
precisely: how do you get a latent space you can actually sample from?
Chapters 9–12 build attention and then use it to replace the CNN in the two
places the earlier chapters left it. Chapters 13–17 are the current frontier:
learning from image–text pairs instead of labels, feeding images to a language
model, questioning whether attention is necessary at all, and the engineering
that decides what actually runs.

## How the notebooks are written

- **Every concept gets three passes**, in this order:
  🧠 **the intuition** (no symbols) → 📐 **the math** (paper-readable) →
  💻 **the code** (executable, in the next cell).
- Modules are separated by `---` and a `# Module N — Title` heading, with
  module numbers in Roman numerals and subsections lettered within their
  module (`II.a`, `II.b`); each chapter opens with a table of its modules and
  closes with a `| You built | the lesson |` wrap-up.
- Tunable cost/quality constants are marked `# <- knob` inline.
- Datasets come from the Hugging Face `datasets` library; models are built from
  scratch rather than imported.

## A note on the framework

**Chapters 1–8 are Keras; chapters 9–17 are PyTorch.** The switch is deliberate:

- The tooling for attention is PyTorch-only. Hugging Face `transformers`
  dropped TensorFlow in v5, `diffusers` never had it, and the reference
  implementations of DiT, SD3, and Flux are all PyTorch. Writing torch means
  you can diff your work against the real thing.
- On an RTX 50-series card it is also simply faster: TensorFlow 2.21 ships no
  `sm_120` kernels, so it JIT-compiles everything from PTX. PyTorch's cu128
  build has native Blackwell kernels.

The mental model transfers directly: `keras.layers.Layer` → `nn.Module`,
`build()`/`call()` → `__init__()`/`forward()`. The one real difference is that
there is no `model.fit()`, so you write the training loop yourself. Chapter 9,
Module IV introduces it line by line.

## Setup

```bash
uv pip install --python <venv>/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python <venv>/bin/python "transformers>=5"
```

`transformers` is used **only** for its tokenizer; every model in these
notebooks is built from scratch.

## Runtimes

| Chapter | Time | Notes |
|---|---|---|
| 1 | ~3 min (CPU) | all data synthetic, nothing downloads |
| 2 | ~15 min (CPU) | first run also downloads ~160 MB of datasets |
| 3 | ~20 min (CPU) | two 32-layer networks raced head-to-head |
| 4 | ~15 min (CPU) | frozen-backbone embeddings, then two-stage fine-tuning |
| 5 | ~1 min GPU / ~8 CPU | one training run |
| 6 | ~3 min GPU / ~15 CPU | the plain-AE baseline plus the VAE |
| 7 | ~7 min (GPU) | 50 epochs over all 60k images; GANs need the epochs |
| 8 | ~6 min (GPU) | two 40-epoch trainings plus the sampling loops |
| 9 | < 1 min (GPU) | one 15-second training run; the rest is small matrices |
| 10 | ~1 min (GPU) | eight short runs (a demo, a head sweep, a seed control) |
| 11 | ~6 min (GPU) | four 30-epoch runs on all 50k CIFAR images |
| 12 | ~13 min (GPU) | a three-way ablation (~5 min) plus the real denoiser (~7 min) |
| 13 | ~3 min (GPU) | one CLIP run, then six more for the batch-size sweeps |
| 14 | ~2 min (GPU) | two pretraining runs, then five VLM ablations |
| 15 | ~6 min (GPU) | nine models on associative recall, plus a scaling benchmark |
| 16 | ~4 min (GPU) | one pretraining run, then a dozen short adaptations |
| 17 | < 1 min (GPU) | nothing is trained: one algorithm and a benchmark |

On CPU, lower `EPOCHS` first; every cost knob is marked `# <- knob`.
Chapters 15 and 17 need a CUDA device for their memory measurements.

## What these notebooks do differently

Every quantitative claim is measured in the notebook rather than asserted, and
**several of the measurements do not match the textbook story.** They are
reported as they came out:

- **Chapter 5** builds the obvious way to sample from an autoencoder's latent
  space and gets grey fog. That failure is why chapters 5–8 exist in the order
  they do.
- **Chapter 10** finds that head count makes no difference to accuracy on IMDB:
  the spread across 1–8 heads is 1.08× the run-to-run noise floor, established
  with a seed control. The heads *do* learn near-uncorrelated patterns; the
  diversity is real, the benefit on this task is not. This agrees with Michel et
  al. (2019) and Voita et al. (2019).
- **Chapter 11** finds the ViT's heads never learn locality at all: mean
  attention distance sits at the uniform-attention baseline in every head of
  every block. That is the direct mechanistic cause of the CNN beating it, and
  it sits alongside positional embeddings which *did* rediscover the 2-D grid.
  Knowing the geometry and exploiting it are separate problems.
- **Chapter 12** attempts to measure the diversity/fidelity trade of
  classifier-free guidance and **fails**: pixel-space distance tracks contrast
  rather than diversity, so the curve comes out backwards. The failure is kept and
  explained.
- **Chapter 13** runs the batch-size sweep under two defensible protocols and
  gets **opposite conclusions**, because batch size moves the negative count,
  the update count, and the data seen all at once. Neither control isolates the
  quantity the folklore is about. It also reproduces the **modality gap**, with
  image and text embeddings occupying separate cones despite being trained to align.
- **Chapter 14** shows a 33,000-parameter projector matching a 4.7M-parameter
  full fine-tune, but only when the language model is pretrained. The
  random-LM control scores zero, which is what makes the result mean anything.
- **Chapter 15** finds selection is worth a large multiple over a
  time-invariant SSM, and that **attention still wins** on recall, because a
  fixed-size state is structurally bad at retrieval. It also finds the parallel
  scan losing on wall-clock while winning on memory.
- **Chapters 16 and 17** measure NF4 quantization and a hand-tiled
  FlashAttention that are algorithmically right and practically slow, landing on
  the same lesson Chapter 15 reached from the other side. Asymptotics are a
  claim about the limit, not a promise about your hardware.

The intent is that you finish having *measured* these architectures rather than
only read about them, including the parts where the measurement disagrees with
the summary.
