Here’s a concrete, end-to-end implementation plan you can literally follow step by step. I’ll assume your project root is:

`~/unsupervised-ired`

and that **all written summaries + report bits go in** `~/unsupervised-ired/documentation`, and **all experiment outputs** (plots, tables, logs) go in `~/unsupervised-ired/documentation/results`.

You can of course tweak names, but I’ll write everything as if that’s the setup.

---

## 0. Project skeleton & environment

**Goal:** Have a clean, repeatable structure so you never wonder “where does this go?”

### 0.1 Create directories

In a terminal:

```bash
mkdir -p ~/unsupervised-ired
cd ~/unsupervised-ired

mkdir -p data/ired
mkdir -p notebooks
mkdir -p src
mkdir -p documentation/report_sections
mkdir -p documentation/results
mkdir -p documentation/reading_notes
```

So:

* **Reading notes** → `documentation/reading_notes/...`
* **Report draft pieces** → `documentation/report_sections/...`
* **All plots / CSV / metrics from code** → `documentation/results/...`

### 0.2 Set up Python environment

Inside the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install numpy scipy scikit-learn matplotlib jupyter ipykernel
# optional diffusion-maps libs; if they fail, you’ll just implement manually
pip install pydiffmap
```

Register kernel (optional, but nice):

```bash
python -m ipykernel install --user --name stat185-ired --display-name "STAT185 IRED"
```

> **Docs requirement:**
> Create a short file documenting the environment:

`documentation/reading_notes/environment_setup.md`
Describe:

* Python version
* Main libraries
* Any gotchas (e.g., how to activate the venv)

---

## 1. Background reading + summaries (put in `~/documentation/reading_notes`)

**Goal:** Understand just enough IRED + manifold learning to write the Overview & Algorithm sections later.

### 1.1 Read IRED (primary paper / site)

**Inputs:**

* IRED paper
* IRED project website (the page you linked)

**Tasks:**

1. Skim once for the big picture.
2. Second pass focusing on:

   * How energy functions are defined.
   * What an “energy trajectory” is (what’s changing each step).
   * What domains they report experiments on (Sudoku, graphs, etc.).
3. Identify **one primary domain** you will use (e.g. Sudoku or matrix inverse).

**Output (required):**

Create:

`documentation/reading_notes/ired_notes.md`

Include:

* 1–2 paragraph summary of IRED’s goal.
* Bullet list of domains they apply to.
* Description of the optimization trajectory for your chosen domain.
* Notes on what information you can realistically log (state x, energy, success, difficulty, etc.).

### 1.2 Read Diffusion Maps / Laplacian Eigenmaps

You don’t need full proofs, just algorithm + intuition.

**Tasks:**

1. For **Diffusion Maps**:

   * Write down how you build the graph.
   * Write down how you form the Markov matrix and get eigenvectors.
   * Note where bandwidth / k-NN choices enter.
2. Optionally skim **Laplacian Eigenmaps** to mention as a related method.

**Output (required):**

Create:

`documentation/reading_notes/diffusion_maps.md`

Sections:

* “Intuition”: 1–2 paragraphs in your own words.
* “Algorithm”: numbered steps (from data → graph → eigenvectors → embedding).
* “Tuning parameters”: list with 1–2 sentences each (k, epsilon/bandwidth, # of eigenvectors, subsampling).
* “How this might apply to IRED trajectories”: 1 short paragraph connecting the two.

If you read about Laplacian Eigenmaps, add it either to that file or to:

`documentation/reading_notes/laplacian_eigenmaps.md`

### 1.3 Start writing report background while it’s fresh

Immediately after those readings, write the **first draft** of your Overview/Background sections.

**Output:**

* `documentation/report_sections/01_background_ebm_ired.md`

  * Explain EBM + IRED in ~0.5–1 page.
* `documentation/report_sections/02_background_manifold_learning.md`

  * Explain manifold learning and Diffusion Maps in ~0.5–1 page.
  * This will later feed directly into your final report.

---

## 2. Getting IRED trajectories (data pipeline)

**Goal:** Extract actual high-dimensional trajectories from the IRED code and save them under `data/ired`.

### 2.1 Clone the IRED repo and explore

Inside project root:

```bash
cd ~/unsupervised-ired
git clone <IRED_REPO_URL> external/ired
```

(Replace with the actual URL from the project’s GitHub.)

Create a quick exploration notebook:

`notebooks/01_explore_ired_code.ipynb`

Tasks in that notebook:

* Inspect where the optimization loop lives.
* Identify:

  * The variable representing the state (x).
  * Where energy is computed.
  * Where the code decides convergence / success.

**Documentation:**

Write:

`documentation/reading_notes/ired_code_structure.md`

with:

* Pointers to the main training/evaluation files.
* Notes like “state is stored in tensor `state` (shape …)” etc.
* The exact functions you’ll hook into for logging.

### 2.2 Implement logging of trajectories

Create a small Python module to hold your logging helpers, e.g.:

`src/trajectory_logging.py`

Example functions:

```python
import numpy as np
from pathlib import Path

def init_trajectory_log():
    return {
        "state": [],
        "energy": [],
        "problem_id": [],
        "step": [],
        "difficulty": [],
        "success": [],
        "landscape_idx": [],
    }

def append_step(log, state, energy, problem_id, step, difficulty, success, landscape_idx):
    log["state"].append(state.copy())  # or state.detach().cpu().numpy()
    log["energy"].append(float(energy))
    log["problem_id"].append(problem_id)
    log["step"].append(step)
    log["difficulty"].append(difficulty)
    log["success"].append(success)
    log["landscape_idx"].append(landscape_idx)

def save_trajectory_log(log, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state=np.stack(log["state"]),
        energy=np.array(log["energy"]),
        problem_id=np.array(log["problem_id"]),
        step=np.array(log["step"]),
        difficulty=np.array(log["difficulty"]),
        success=np.array(log["success"]),
        landscape_idx=np.array(log["landscape_idx"]),
    )
```

In the IRED evaluation code for your chosen domain:

* Initialize `log = init_trajectory_log()`.
* In each optimization step, call `append_step(...)`.
* After finishing a batch of problems, call:

```python
from pathlib import Path
from trajectory_logging import save_trajectory_log

save_trajectory_log(
    log,
    Path("data/ired/sudoku_trajectories_run1.npz")
)
```

**Documentation:**

After you get logging working **for even one run**, write:

`documentation/reading_notes/trajectory_logging_design.md`

* Format of your `.npz` file.
* Shape of `state`.
* Any limitations (e.g., max steps, memory).

### 2.3 Collect a usable dataset

Decide on a modest size for the project:

* e.g. **100–300 Sudoku instances** × **50 steps** each.

Run the instrumented IRED code and save:

* `data/ired/sudoku_trajectories_run1.npz`
* optionally `..._run2.npz` if needed.

**Results (required):**

* In a quick notebook `notebooks/02_summarize_trajectories.ipynb`, load the `.npz`:

  * Check shapes, ranges of energy, success rates.

  * Save a simple summary CSV to:

    `documentation/results/02_trajectory_summary.csv`

  * Save 1–2 quick plots:

    * energy vs step for a few random trajectories → `documentation/results/02_energy_vs_step_examples.png`

* Write a 1–2 paragraph note:

  `documentation/reading_notes/trajectory_dataset_overview.md`
  describing:

  * How many problems
  * How many steps
  * Fraction solved
  * Any weirdness

---

## 3. Implement manifold learning (PCA + Diffusion Maps)

**Goal:** Have code that can take a matrix of states and produce low-dim embeddings.

### 3.1 Preprocess the states

Notebook:

`notebooks/03_preprocess_states.ipynb`

Tasks:

1. Load `sudoku_trajectories_run1.npz`.
2. Flatten each state to 1D, make a matrix `X` of shape `(N_points, D)`, where `N_points` = number of (problem, step) pairs.
3. Standardize features (e.g., zero mean, unit variance) with `sklearn.preprocessing.StandardScaler`.

**Results:**

* Save preprocessed data (optional) to:

  * `data/ired/sudoku_states_flattened.npz`
* Write a brief description of preprocessing choices in:

  `documentation/reading_notes/preprocessing_choices.md`

### 3.2 PCA baseline

In the same notebook or `notebooks/04_pca_baseline.ipynb`:

* Run PCA to 2D / 3D.

* Plot:

  * PCA(1) vs PCA(2), colored by step.
  * PCA(1) vs PCA(2), colored by energy.

* Save plots to:

  * `documentation/results/04_pca_step_colored.png`
  * `documentation/results/04_pca_energy_colored.png`

* Save explained variance ratios to a CSV:

  * `documentation/results/04_pca_explained_variance.csv`

**Docs:**

Short note:

`documentation/reading_notes/pca_baseline_observations.md`

with 3–5 bullet points about what you see.

### 3.3 Diffusion Maps implementation

Create a reusable module:

`src/diffusion_maps.py`

Implement something like:

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def diffusion_maps(X, n_components=3, k=20, epsilon=None, alpha=0.5):
    # X: (N, D)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nn.kneighbors(X)
    
    if epsilon is None:
        epsilon = np.median(distances[:, -1]) ** 2
    
    N = X.shape[0]
    row_idx = np.repeat(np.arange(N), k)
    col_idx = indices.ravel()
    dist_sq = (distances ** 2).ravel()
    weights = np.exp(-dist_sq / epsilon)
    
    W = csr_matrix((weights, (row_idx, col_idx)), shape=(N, N))
    W = 0.5 * (W + W.T)  # symmetrize
    
    d = np.array(W.sum(axis=1)).flatten()
    D_inv = csr_matrix((1.0 / d, (np.arange(N), np.arange(N))), shape=(N, N))
    
    K = D_inv @ W  # Markov matrix
    
    # optionally apply alpha-normalization here if you want
    vals, vecs = eigsh(K.T, k=n_components + 1, which='LM')
    
    idx = np.argsort(-vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # skip the first trivial eigenvector
    lambdas = vals[1:n_components+1]
    psi = vecs[:, 1:n_components+1]
    
    return lambdas, psi
```

Notebook:

`notebooks/05_diffusion_maps_on_sudoku.ipynb`

Tasks:

* Load `X`.
* Run `diffusion_maps` with a few k, epsilon choices.
* For a chosen configuration, save:

  * 2D scatter of ψ1 vs ψ2 colored by:

    * step → `documentation/results/05_dmaps_step_colored.png`
    * energy → `documentation/results/05_dmaps_energy_colored.png`
    * difficulty or success → `documentation/results/05_dmaps_success_colored.png`
* Save eigenvalues to:

  * `documentation/results/05_dmaps_eigenvalues.csv`

**Docs:**

* `documentation/reading_notes/dmaps_implementation_notes.md`
  Describe:

  * The final hyperparameters you picked (k, epsilon, #components).
  * Any numerical issues.

* Start drafting Algorithm section:

  `documentation/report_sections/03_algorithm_diffusion_maps.md`
  Put your cleaned-up description of the algorithm with equations and pseudo-code.

---

## 4. Experiments (toy + IRED) and results logging

You want **two examples** for the report:

1. A toy, controllable energy landscape.
2. IRED trajectories.

### 4.1 Toy energy landscape experiment

Notebook:

`notebooks/06_toy_energy_landscape.ipynb`

Tasks:

1. Define a 2D energy function like a double-well:

   [
   E(x, y) = (x^2 - 1)^2 + 0.1 y^2
   ]

2. Sample several starting points.

3. Run gradient descent with noise for T steps to generate trajectories.

4. Apply:

   * PCA (for sanity).
   * Diffusion Maps to the 2D points (they should recover a 1D coordinate along the wells).

**Results (save all):**

* Energy landscape contour plot:

  * `documentation/results/06_toy_energy_contours.png`
* Trajectories in original space:

  * `documentation/results/06_toy_trajectories_xy.png`
* Diffusion Maps embedding:

  * `documentation/results/06_toy_dmaps_embedding.png`
* Any metrics (eigenvalues, etc.) to CSV:

  * `documentation/results/06_toy_dmaps_eigenvalues.csv`

**Docs:**

* `documentation/reading_notes/toy_experiment_observations.md`

  * 1–2 paragraphs on what the toy example shows about diffusion maps.

* Draft Example 1 section:

  `documentation/report_sections/04_example1_toy_landscape.md`

### 4.2 IRED Sudoku (or chosen domain) experiment

Notebook:

`notebooks/07_ired_sudoku_manifold.ipynb`

Tasks:

1. Load the preprocessed IRED states `X` and metadata (energy, step, problem_id, success, difficulty).
2. Run Diffusion Maps using your chosen hyperparameters.
3. For each problem:

   * Plot the trajectory in (ψ1, ψ2) ordered by step, with a line connecting points.
   * Compare successful vs failed problems.
4. Investigate:

   * Do trajectories look like smooth curves?
   * Does energy generally decrease along a particular diffusion coordinate?
   * Are hard problems further away from a “solution region” in the diffusion space?

**Results (save all plots/metrics into `documentation/results`):**

Examples:

* `documentation/results/07_ired_dmaps_traj_examples.png`
  (a grid of a few trajectories in ψ1–ψ2)
* `documentation/results/07_ired_dmaps_energy_vs_psi1.png`
  (scatter with energy vs ψ1)
* `documentation/results/07_ired_traj_length_vs_difficulty.csv`
  (table of path length in embedding vs difficulty)

**Docs:**

* `documentation/reading_notes/ired_experiment_observations.md`

  * Bullets: relations between energy, difficulty, and geometry.
* Draft Example 2 section:

  `documentation/report_sections/05_example2_ired_trajectories.md`

---

## 5. Analysis, interpretation, and report assembly

Now you have methods + experiments + results. Time to turn it into a coherent report while continuing to log everything.

### 5.1 Synthesize observations

Create a short synthesis notebook:

`notebooks/08_analysis_summary.ipynb`

Tasks:

* Load key CSVs from `documentation/results`.
* Compute any final summary metrics (e.g., correlation between energy and diffusion coordinate, average path length vs difficulty).
* Save a final “metrics table”:

  `documentation/results/08_final_metrics_summary.csv`

**Docs:**

Write:

`documentation/reading_notes/overall_findings.md`

with:

* 3–5 key findings (numbered).
* 2–3 limitations.
* 2–3 ideas for future work.

### 5.2 Draft remaining report sections

Use your existing section files and add new ones:

* **Related work / context:**

  * `documentation/report_sections/06_related_work.md`
* **Discussion & limitations:**

  * `documentation/report_sections/07_discussion_and_limitations.md`
* **Conclusion:**

  * `documentation/report_sections/08_conclusion.md`

In each, explicitly reference:

* Figures: “see Figure X in `documentation/results/...`”.
* Your metrics table from `08_final_metrics_summary.csv`.

### 5.3 Assemble full report

Create a single combined draft:

`documentation/report_sections/full_report_draft.md`

Copy in order:

1. `01_background_ebm_ired.md`
2. `02_background_manifold_learning.md`
3. `03_algorithm_diffusion_maps.md`
4. `04_example1_toy_landscape.md`
5. `05_example2_ired_trajectories.md`
6. `06_related_work.md`
7. `07_discussion_and_limitations.md`
8. `08_conclusion.md`

You can either:

* Concatenate manually, or
* Write a small script `src/assemble_report.py` that reads them in order and writes `full_report_draft.md`.

Example script:

```python
from pathlib import Path

root = Path("documentation/report_sections")
order = [
    "01_background_ebm_ired.md",
    "02_background_manifold_learning.md",
    "03_algorithm_diffusion_maps.md",
    "04_example1_toy_landscape.md",
    "05_example2_ired_trajectories.md",
    "06_related_work.md",
    "07_discussion_and_limitations.md",
    "08_conclusion.md",
]

with open(root / "full_report_draft.md", "w") as out:
    for fname in order:
        path = root / fname
        out.write(open(path).read())
        out.write("\n\n")
```

You’ll then convert `full_report_draft.md` into whatever format the class expects (LaTeX / Word / PDF).

---

## 6. Final checklist (quick reference)

You’re “done” when:

1. **Reading + notes**

   * [ ] `documentation/reading_notes/ired_notes.md`
   * [ ] `documentation/reading_notes/diffusion_maps.md`
   * [ ] `documentation/reading_notes/trajectory_dataset_overview.md`
   * [ ] `documentation/reading_notes/preprocessing_choices.md`
   * [ ] `documentation/reading_notes/pca_baseline_observations.md`
   * [ ] `documentation/reading_notes/dmaps_implementation_notes.md`
   * [ ] `documentation/reading_notes/toy_experiment_observations.md`
   * [ ] `documentation/reading_notes/ired_experiment_observations.md`
   * [ ] `documentation/reading_notes/overall_findings.md`

2. **Code + data**

   * [ ] IRED repo cloned under `external/ired`.
   * [ ] Logging hooked into one domain.
   * [ ] At least one `.npz` trajectories file under `data/ired/`.
   * [ ] PCA + Diffusion Maps implemented and tested.
   * [ ] Toy energy experiment notebook completed.

3. **Results (must all live in `documentation/results`)**

   * [ ] Basic trajectory summaries.
   * [ ] PCA plots.
   * [ ] Diffusion Maps eigenvalues + embeddings plots.
   * [ ] Toy experiment plots.
   * [ ] IRED embedding plots.
   * [ ] Final metrics CSV.

4. **Report sections**

   * [ ] Background (EBM/IRED).
   * [ ] Background (manifold learning).
   * [ ] Algorithm (Diffusion Maps).
   * [ ] Example 1 (toy).
   * [ ] Example 2 (IRED).
   * [ ] Related work.
   * [ ] Discussion + limitations.
   * [ ] Conclusion.
   * [ ] `full_report_draft.md` assembled.
