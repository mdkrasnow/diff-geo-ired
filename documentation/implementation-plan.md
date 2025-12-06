Here’s a concrete, “do this step by step” implementation plan that gets you from **idea → math understanding → experiments → finished paper**, using the `~/documentation` folder as the hub.

I’ll assume a rough 2–3 week window; if you have less time, you can compress.

---

## 0. Project skeleton

**Goal:** Set up a clean structure so everything you read, code, and discover feeds into the final paper.

### 0.1 Create directory layout

In your project root (wherever you’re working for this class):

```bash
mkdir -p ~/documentation
mkdir -p ~/documentation/results
mkdir -p ~/documentation/notes
mkdir -p ~/documentation/drafts
mkdir -p ~/documentation/figures
```

You’ll use:

* `~/documentation/notes` for reading summaries and theory notes
* `~/documentation/results` for CSVs, NumPy dumps, logs, etc.
* `~/documentation/figures` for plots, images, screenshots (with credits)
* `~/documentation/drafts` for paper drafts (sections, outline, final)

---

## 1. Lock in topic, case study, and outline

**Goal:** A precise research question and a rough paper structure so you’re not wandering.

### 1.1 Decide the core angle (write it down)

Create `~/documentation/notes/project_overview.md` and write:

* 1–2 paragraphs on:

  * Your topic:

    > “A differential-geometric view of IRED: energy trajectories as curves on learned manifolds.”
  * Your **case study**:

    * Pick *one*:

      * **Matrix inverse** (continuous, nice Euclidean geometry)
      * **Planning on graphs** (nice discrete manifold / graph geometry)
  * The main question in one sentence, e.g.:

    > “How do IRED’s optimization trajectories behave as curves on a manifold, and what does manifold learning reveal about their geometry?”

### 1.2 Draft a rough paper outline

In the same file or a new one: `~/documentation/drafts/outline.md`

Outline something like:

1. Introduction and motivation
2. Background: differential geometry and manifold learning
3. IRED: energy diffusion and iterative reasoning
4. Case study: geometry of energy trajectories (your chosen task)
5. Discussion (intrinsic vs extrinsic, curvature, etc.)
6. Conclusion and future directions

This is your target; the rest of the plan feeds into these sections.

---

## 2. Theory reading + writeups (course-aligned)

**Goal:** Refresh the specific diff-geo and manifold learning tools you’ll actually use in the paper, and capture them in your own words.

### 2.1 Differential geometry refresher

Create `~/documentation/notes/diff_geo_background.md`

Read your course notes / textbook on:

* Curves in (\mathbb{R}^n)

  * Parametrized curves, arc length, unit-speed reparametrization
  * Curvature (\kappa) and torsion (only if you need it)
* Riemannian manifolds ((M,g))

  * Tangent space, metric, length and energy of curves
* Geodesics and Euler–Lagrange

  * Geodesic equation (with Christoffel symbols)
  * Interpretation of geodesics vs gradient flow

**Write in that note file:**

1. Clear definitions (your own words):

   * Curve, arc length, energy functional
   * Geodesic vs general curve
2. One short worked example:

   * Show that straight lines in (\mathbb{R}^n) are geodesics.
3. 1–2 paragraphs linking to your project:

   * E.g.

     > “In this project, I treat IRED trajectories as discrete analogues of gradient flow curves on a manifold, not geodesics. Geodesics extremize length, while gradient flow curves maximize rate of energy decrease.”

### 2.2 Manifold learning background

Create `~/documentation/notes/manifold_learning_background.md`

Read slides / a survey on:

* The **manifold hypothesis**
* Basic algorithms:

  * Isomap (geodesic distance on a kNN graph)
  * LLE (local linear reconstructions)
  * Laplacian Eigenmaps (graph Laplacian, eigenvectors)

**Write:**

1. 1–2 paragraphs per method:

   * What it does.
   * What geometry it tries to preserve (geodesic distances, local linear structure, eigenfunctions of Laplace–Beltrami).
2. A short “bridge paragraph”:

   * How these methods could help **visualize IRED trajectories** in low dimensions.
   * E.g., “We use manifold learning to embed high-dimensional IRED states into 2D, to inspect whether trajectories follow smooth paths on a low-dimensional manifold.”

### 2.3 IRED paper + website understanding

Create `~/documentation/notes/ired_method_summary.md`

Read:

* The IRED paper (focus: method, energy formulation, energy diffusion)
* The IRED website’s visualizations for your chosen task.

For this file, write:

1. A clean definition of:

   * (E_\theta(x,y,k)): what each variable means
   * What the “landscape index” / diffusion schedule is doing
   * The update rule at inference time, schematically
2. A description of **your chosen case study**:

   * Inputs, outputs, what counts as a correct solution
   * What the visualizations are showing (e.g. error maps getting cleaner)
3. A bullet list of “geometric interpretations”:

   * State space as manifold or discrete manifold
   * Energy as scalar field
   * Optimization path as discrete gradient flow
   * Early vs late landscapes = changing curvature / shape of the potential

---

## 3. Experimental design and logging plan

**Goal:** Decide exactly what data you’ll collect from IRED and what geometric / manifold-learning analyses you’ll run so you don’t end up with random plots.

Create `~/documentation/notes/experiment_design.md`

### 3.1 Define the dataset you’ll log

Examples (pick one set):

* **If matrix inverse:**

  * For each random input matrix (A), IRED iteratively updates a candidate inverse (B_t).
  * You log **all (B_t)** across time and landscapes.
* **If planning on graphs:**

  * For each problem, you log the vector of node scores (s_t) at each iteration.

Specify in the note:

* Shape and type of your state:

  * e.g., flattened matrix (B_t \in \mathbb{R}^{n^2}).
* What you’ll log:

  * `state` (high-dimensional vector)
  * `energy` value
  * `step index t`
  * `landscape index k`
  * maybe error metric versus ground truth if available

### 3.2 Choose manifold learning / analysis methods

In the same file, write:

* “Core analyses I will perform”:

  1. Basic **PCA** for sanity check embeddings.
  2. One **nonlinear manifold learning method** (Isomap or Laplacian Eigenmaps) for 2D embeddings of all states.
  3. Simple geometric diagnostics:

     * Trajectory length in embedding space
     * A discrete curvature proxy:
       [
       \kappa_t \approx \frac{|y_{t+1} - 2y_t + y_{t-1}|}{|y_{t+1} - y_t|^2}
       ]
       (if you have time).

Tie each to a figure you’ll want in the paper, e.g.:

* Fig 1: Sample 2D embedding with trajectories colored by time.
* Fig 2: Energy vs step index.
* Fig 3: Histogram of discrete curvature along trajectories.

---

## 4. Code + data collection

**Goal:** Run IRED for your case study, log the trajectories, and store all raw data in `~/documentation/results`.

### 4.1 Environment + baseline runs

Create a note: `~/documentation/notes/code_setup.md`

In that file, track:

* Git repo you cloned
* Python version and libraries
* Exact commands you used to:

  * Install dependencies
  * Run a baseline example for your chosen task

Example commands (adapt to your env):

```bash
# inside your project (not literally ~/documentation)
git clone <ired-repo-url> ired-project
cd ired-project
# create virtual env, install deps...
```

After you get a baseline example working:

* Write a short “Sanity check” section in `code_setup.md`:

  * Which example you ran
  * What outputs you saw (e.g., matrix error decreased, Sudoku solved)

### 4.2 Implement logging of trajectories

Create a Python script or notebook, e.g. `log_trajectories.py` or `ired_trajectories.ipynb`, that:

1. Runs IRED on multiple problem instances (e.g. 50–200).

2. At each iteration and for each landscape index:

   * Extracts the current state (y_t) (flattened vector).
   * Records:

     * `problem_id`
     * `step`
     * `landscape`
     * `state` (array)
     * `energy` (scalar)
     * any error metric

3. Saves logs to `~/documentation/results` as:

   * `ired_trajectories_raw.npz` **or**
   * `ired_trajectories.csv` (if the dimension is manageable)

Document this in `~/documentation/notes/experiment_design.md`:

* File names
* Number of trajectories
* Dimensions

---

## 5. Manifold learning + geometric analysis

**Goal:** Turn raw trajectories into embeddings, plots, and simple geometric measurements. Save all results into `~/documentation/results` and `~/documentation/figures`.

### 5.1 Embeddings and visualizations

Create a notebook: `ired_embedding_analysis.ipynb`

In that notebook:

1. **Load trajectories** from `~/documentation/results/ired_trajectories_raw.*`.
2. For each problem, or for a subset:

   * Stack all states (y_t) into a matrix `X` (rows = time steps; columns = features).
3. Run PCA on `X` to get a 2D embedding:

   * Save:

     * `pca_embedding.npy` to `~/documentation/results`
     * Plot of trajectories in 2D:

       * Points connected in time order
       * Colored by step or landscape index
       * Save as `~/documentation/figures/pca_trajectories_caseX.png`
4. Run a nonlinear method (Isomap or Laplacian Eigenmaps) on the same data:

   * Save embeddings and plots similarly.

Along the way, export quantitative results (e.g., positions, energies) as CSV/NPZ into `~/documentation/results`.

### 5.2 Simple geometric diagnostics

In the same notebook or a new one:

* For each trajectory:

  * Compute **discrete path length** in embedding space:
    [
    L = \sum_t |z_{t+1} - z_t|
    ]
  * Compute a simple **discrete curvature proxy** if feasible:

    * Skip the endpoints, compute (\kappa_t) for interior points.
* Store these metrics as:

  * `ired_trajectory_lengths.csv`
  * `ired_trajectory_curvatures.csv`
    in `~/documentation/results`.

Create a short note: `~/documentation/notes/results_summary.md` where you describe:

* What you observed:

  * Do trajectories look smooth?
  * Do they cluster near low-dimensional curves or tubes?
  * Do early vs. late landscapes show different behavior?
* Reference specific figures and result files.

---

## 6. Progressive paper writing (section by section)

**Goal:** Don’t leave writing for the end. Use your notes to progressively assemble the final expository paper.

### 6.1 Background sections

Create `~/documentation/drafts/background.md`

Fill it with:

1. **Differential geometry background**:

   * Rework your `diff_geo_background.md` into a clean narrative.
   * Use your course notation and emphasize:

     * Curves, geodesics, gradient flow.
2. **Manifold learning background**:

   * Rework `manifold_learning_background.md` into 1–1.5 pages.
3. **EBMs and IRED**:

   * Rework `ired_method_summary.md` into a formal description.

When you write, keep in mind grading items:

* **Mathematical correctness**: be precise with definitions and equations.
* **Clarity**: short paragraphs, clear sentences.
* **Adaptation to course**: explicitly reference concepts from your syllabus (first fundamental form, Riemannian metric, etc., even if briefly).

### 6.2 Methods section (your geometric lens + experimental design)

Create `~/documentation/drafts/methods_case_study.md`

Include:

1. How you interpret the state space as a manifold / discrete manifold.
2. How IRED’s update rule corresponds to discrete gradient flow.
3. The data you logged (as designed in Step 3).
4. The manifold learning / embedding pipeline:

   * PCA + Isomap/Laplacian Eigenmaps.
   * The metrics you compute (length, curvature).

Make sure you point to the specific result files in `~/documentation/results` so it’s easy to verify later.

### 6.3 Results and discussion sections

Create `~/documentation/drafts/results_and_discussion.md`

Use:

* Figures from `~/documentation/figures`
* Summaries from `~/documentation/notes/results_summary.md`

Structure:

1. **Qualitative results**:

   * Show embeddings and describe:

     * Do trajectories follow smooth low-dimensional paths?
     * How energy changes along trajectories.
2. **Quantitative results**:

   * Report length and curvature statistics.
3. **Geometric interpretation**:

   * Connect:

     * Gradient flow on manifolds vs what you see.
     * Intrinsic/extrinsic geometry: how much your observations depend on the embedding.
4. **Connections to deep theorems**:

   * Briefly mention:

     * Intrinsic vs extrinsic (Theorema Egregium) as a conceptual analogy.
     * Discrete manifold view if you used a graph-based case.

---

## 7. Final assembly and polishing

**Goal:** Merge drafts into one coherent 4+ page paper, add references, and check against grading criteria.

### 7.1 Combine drafts into a single document

Create `~/documentation/drafts/final_report.md` (or `.tex` if you prefer LaTeX)

Combine, in order:

1. New **Introduction** (1 page)

   * Motivation.
   * Problem statement.
   * Brief overview of what’s in the rest of the paper.
2. `background.md` (trimmed + smoothed).
3. `methods_case_study.md`.
4. `results_and_discussion.md`.
5. **Conclusion** (0.5 page)

   * What you learned.
   * Limitations.
   * Future directions (e.g., more advanced Riemannian optimization, other tasks).
6. **References**.

### 7.2 References and acknowledgements

In `final_report.md`, add a **References** section:

* IRED paper.
* One or two manifold learning references.
* Your diff-geo text / notes.
* Any additional EBMs papers if you cite them.

Add an **Acknowledgements / Tools** note:

* Acknowledge:

  * The IRED GitHub repo / authors.
  * Any software used (NumPy, scikit-learn, etc.).
  * Use of “computer algebra / Python / Jupyter notebooks.”
  * Optionally: that you used an AI assistant for planning but did your own reading, coding, and writing.

### 7.3 Check against grading criteria

In a final small note `~/documentation/notes/self_check.md`, make a checklist:

1. **Mathematical correctness**

   * Are definitions of geodesics, gradient flow, manifold learning methods correct?
2. **Clarity, readability, elegance**

   * Are sections well-structured?
   * Are figures clearly labeled and referenced?
3. **References and sources**

   * Are all external ideas cited?
4. **Adaptation to the course**

   * Do you explicitly connect to notions like:

     * Curves and surfaces
     * Riemannian metric
     * Geodesics and curvature
     * Discrete manifolds / Gauss–Bonnet (even if only in passing)?
5. **Originality / depth / surprise**

   * Do you offer a genuine geometric *interpretation* of IRED, not just a summary?
   * Does your manifold-learning analysis go beyond what the original paper shows?
