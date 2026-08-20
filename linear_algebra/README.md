# Hands-On Linear Algebra

Nine standalone notebooks that build linear algebra from the definition of a
vector up to the singular value decomposition, with every concept explained three
times (intuition, math, code) and every quantitative claim **measured in the
notebook rather than asserted**.

Written for someone who took calculus, and possibly linear algebra, some years
ago: nothing is assumed beyond comfort with functions and a little Python. The
course is self-contained and strictly cumulative, so each chapter opens on the
question the previous one left hanging.

## The chapters, in order

| # | Notebook | The question it answers | Dataset |
|---|---|---|---|
| 1 | [ch01_vectors_and_vector_spaces.ipynb](ch01_vectors_and_vector_spaces.ipynb) | What *is* a vector? The eight axioms, tested on real spaces, a disguised one, and a fake | `ylecun/mnist` |
| 2 | [ch02_span_and_linear_independence.ipynb](ch02_span_and_linear_independence.ipynb) | What can a set of vectors reach, and which of them are redundant? | `ylecun/mnist` |
| 3 | [ch03_basis_and_coordinates.ipynb](ch03_basis_and_coordinates.ipynb) | What is the minimal description of a space, and does the choice of description matter? | `uoft-cs/cifar10`, `ylecun/mnist` |
| 4 | [ch04_linear_transformations.ipynb](ch04_linear_transformations.ipynb) | Where do matrices come from? (From what a map does to a basis) | `ylecun/mnist` |
| 5 | [ch05_solving_linear_systems.ipynb](ch05_solving_linear_systems.ipynb) | When does $A\mathbf{x} = \mathbf{b}$ have no / one / infinitely many solutions? | — |
| 6 | [ch06_determinants_and_inverses.ipynb](ch06_determinants_and_inverses.ipynb) | When can a transformation be undone, and how do you tell? | — |
| 7 | [ch07_orthogonality_and_least_squares.ipynb](ch07_orthogonality_and_least_squares.ipynb) | What do you do when there is no solution? | `ylecun/mnist` |
| 8 | [ch08_eigenvalues_and_eigenvectors.ipynb](ch08_eigenvalues_and_eigenvectors.ipynb) | Is there a basis in which a square matrix becomes trivial? | — |
| 9 | [ch09_svd_and_pca.ipynb](ch09_svd_and_pca.ipynb) | Is there one for *every* matrix, and for data? | `uoft-cs/cifar10`, `ylecun/mnist` |

Every notebook is standalone (it re-imports and redefines what it needs), so you
can drop into any chapter, but the ideas only make sense in order.

## How the notebooks are written

Every concept gets **three passes**, marked
🧠 **The intuition** (no symbols) → 📐 **The math** (paper-readable) →
💻 **The code** (executable, in the next cell). Modules are separated by `---`
and a `# Module N — Title` heading, with module numbers in Roman numerals and
subsections lettered within their module (`II.a`, `II.b`), opened with a
table of what you will build, and closed with a
`| You built | The transferable lesson |` wrap-up. Tunable
constants carry an inline `# <- knob`: sample sizes, matrix sizes, polynomial
degrees, how many components to keep.

Algorithms are built from scratch before the library call, including elimination,
RREF, null-space bases, matrix inverses, cofactor and elimination determinants,
power iteration, Gram–Schmidt, and the SVD itself. Only then is the result
checked against NumPy. Images are pulled live from the Hugging Face Hub, so every
claim about high-dimensional behaviour is made against actual high-dimensional
data rather than a toy matrix.

## What these notebooks do differently

Every number is computed in the notebook, and several of them come out smaller
or messier than the textbook story. Those results are reported as they came out:

- **Chapter 1** blends one digit into another along a straight line, and the
  middle frames are a *double exposure* rather than a digit. The pixel space is
  a vector space; the set of images-of-digits is not a subspace, and the course
  is honest from the first chapter about where linearity stops.
- **Chapter 2** finds that 500 handwritten threes have rank **485**, so they are
  not even formally independent, and that the measured rank swings from 17 to 485
  depending only on the tolerance you pick.
- **Chapter 3** shows the cosine basis beating the pixel basis by **19×** on a
  photograph and by barely **2×** on a handwritten digit, because digits are
  already sparse in pixels. There is no universally best basis.
- **Chapter 6** demonstrates that `det` is **not** a singularity detector: a
  matrix with determinant $10^{-20}$ and condition number 1 sits next to one
  with determinant $10^{-78}$ that is genuinely dangerous.
- **Chapter 7** measures classical Gram–Schmidt losing orthogonality completely
  by $n = 8$, and the normal equations, derived two modules earlier, losing
  most of the available precision on an ill-conditioned fit.
- **Chapter 9** does the storage accounting that image-compression demos of the
  SVD usually omit: on a 32×32 image, rank-$k$ truncation breaks even at
  $k \approx 16$, so per-image SVD is a **poor** compressor. And PCA beats the
  cosine basis on held-out digits by a factor of 1.3–1.7, a real win, but not
  the order of magnitude the demo usually implies.

## Setup

Python 3.13 with four packages:

```bash
uv pip install --python <venv>/bin/python numpy matplotlib datasets ipykernel
```

No GPU, no deep-learning framework; everything here is NumPy. The datasets are
downloaded once from the Hugging Face Hub (MNIST ~11 MB, CIFAR-10 ~170 MB) and
cached locally afterwards.

## Runtimes

All nine run on CPU, and all nine are fast: measured end to end with the
datasets already cached, every chapter finished in **under 6 seconds**, the
range being 1.5 s (Chapter 5) to 5.5 s (Chapter 9, the heaviest). First runs add
the dataset downloads.
