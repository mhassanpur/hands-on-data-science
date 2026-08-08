# Hands-On Data Science

This repo is a collection of hands-on data science courses covering a variety of subjects.

Each course here is a sequence of Jupyter notebooks that builds its subject from
the ground up. The algorithm is written from scratch first and the library call
comes second, so `np.linalg.svd` and `nn.MultiheadAttention` show up as checks on
your own implementation rather than as the lesson. Every quantitative claim is
measured in the notebook, and where a measurement disagrees with the usual
textbook story, the measurement is what gets reported.

## Courses

| Course | Chapters | Stack | Hardware |
|---|---|---|---|
| [Hands-On Linear Algebra](linear_algebra/) | 9 | NumPy | CPU, seconds per chapter |
| [Deep Learning, Built From Scratch](deep_learning/) | 17 | Keras, then PyTorch | CPU for ch. 1–4, CUDA GPU from ch. 5 on |

### [Hands-On Linear Algebra](linear_algebra/README.md)

Nine notebooks that run from the definition of a vector to the singular value
decomposition, by way of span, basis, linear maps, elimination, determinants,
orthogonality, and eigenvectors. Elimination and RREF, null-space bases, matrix
inverses, cofactor and elimination determinants, power iteration, Gram–Schmidt,
and the SVD are each built by hand before NumPy is consulted. Images come live
from the Hugging Face Hub, so claims about high-dimensional geometry are tested
against real high-dimensional data instead of a 3×3 toy matrix. The prerequisite
is comfort with functions and a little Python; the whole course is NumPy and
runs on a laptop.

Start at [ch01_vectors_and_vector_spaces.ipynb](linear_algebra/ch01_vectors_and_vector_spaces.ipynb).

### [Deep Learning, Built From Scratch](deep_learning/README.md)

Seventeen notebooks that open with a single neuron trained by hand-derived
gradients and close on diffusion transformers, vision-language models, state
space models, and the kernel engineering that decides what fits on one GPU. The
four parts are supervised learning (ch. 1–4), generative models (ch. 5–8),
attention (ch. 9–12), and the multimodal and systems frontier (ch. 13–17).
Chapters 1–8 use Keras and chapters 9–17 use PyTorch, a switch the course
README explains and walks you through. Nothing is imported before it has been
built at least once, and `transformers` appears only for its tokenizer.

Start at [ch01_intro_neural_networks.ipynb](deep_learning/ch01_intro_neural_networks.ipynb).

## Conventions shared by every course

Notebooks explain each concept three times in a fixed order: 🧠 the intuition
with no symbols, 📐 the math in a form you could read on paper, then 💻 the code
in the next cell. Modules within a chapter are separated by `---` and a
`# Module N — Title` heading, each chapter opens with a table of the modules it
contains, and each one closes with a two-column wrap-up pairing what you built
against the transferable lesson. Constants you can tune for speed or quality
carry an inline `# <- knob`, which is the first thing to lower if a chapter runs
long on your machine.

Chapters are standalone in the sense that each re-imports and redefines what it
needs, so you can open any one of them and run it top to bottom. The ideas are
cumulative, though, and each chapter opens on the question the previous one left
hanging, so the order is the argument.

## Setup

Python 3.13. [requirements.txt](requirements.txt) pins the full environment for
every course at once:

```bash
uv pip install --python <venv>/bin/python -r requirements.txt
```

`torch` and `torchvision` must come from the PyTorch CUDA 12.8 index rather than
PyPI to get native `sm_120` (Blackwell, RTX 50-series) kernels:

```bash
uv pip install --python <venv>/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

If you only want the linear algebra course, `numpy`, `matplotlib`, `datasets`,
and `ipykernel` are enough, and no GPU is involved. Datasets are pulled from the
Hugging Face Hub on first use (MNIST is about 11 MB, CIFAR-10 about 170 MB) and
cached locally afterwards. Per-chapter runtimes live in each course README.

## Adding a course

A course is one top-level directory holding its chapters and its own README, and
nothing outside that directory needs to change apart from the table above. Name
the directory after the subject in `snake_case`, name chapters
`chNN_topic_in_snake_case.ipynb` starting at `ch01`, and follow the three-pass
structure and `# <- knob` marking described above so a reader moving between
courses does not have to relearn the layout.

The course README carries the detail the root one deliberately omits: a table of
chapters with what each builds and which datasets it touches, the setup specific
to that course, measured runtimes, and an account of the results that came out
against expectation. Add any new dependencies to the shared
[requirements.txt](requirements.txt) with a comment naming the course that needs
them, then add one row to the course table and one section above.

## License

MIT. See [LICENSE](LICENSE).
