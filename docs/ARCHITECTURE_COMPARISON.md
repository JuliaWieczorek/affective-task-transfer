# Interpreting architecture and task-transfer comparisons

## Primary, pre-test comparison

Study II asks which system best supports **emotion-conditioned intensity** under
the fixed BERT-base-uncased protocol. The primary test measure is intensity
emotion-macro-F1. Examine medium/high recall and the joint emotion-intensity
pair score before describing a system as useful. Report emotion and sentiment
scores as task trade-offs, together with trainable parameters, time and peak
VRAM. Checkpoints are selected on the development set by the mean active-task
macro-F1, so a test-time intensity ranking is conditional on balanced
checkpoint selection. Do not change that rule after viewing test scores.

`architecture_deltas.csv` and `architecture_summary.csv` contain matched-seed
contrasts among hard sharing, soft sharing, adapters and MMoE for the *same*
dataset, backbone and task set. They compare complete systems with different
capacity; they do not isolate sharing architecture from parameter count. The
five-seed t intervals describe training variation conditional on one split.
If several systems are indistinguishable within this evidence, report a
trade-off or Pareto set rather than an absolute winner. The literature treats
multi-task performance as a multi-objective trade-off: [Sener and Koltun,
NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html).

## What task contrasts identify

Matched STL vs MTL and the soft-sharing pair-to-triple comparisons estimate
the **net effect of a training configuration** on each task in this protocol.
Adding a task changes its supervision, checkpoint selection and sometimes
capacity. Thus an intensity gain with emotion supervision supports an
observed positive task-composition effect, but does not by itself show a
directional flow of semantic knowledge. [Standley et al., ICML
2020](https://proceedings.mlr.press/v119/standley20a.html) make the importance
of task grouping and computation/accuracy trade-offs explicit.

Current loss trajectories, fixed-probe gradient norms/cosines and MMoE gate
weights can test whether optimisation conflict or gate collapse is *consistent*
with outcomes. Gradient measurements refer to the shared projection in soft
sharing and the shared encoder in the other primary architectures. Their raw
norms are therefore not a common cross-architecture scale. Gate use does not
establish that an expert contributes useful information. [GradNorm, ICML
2018](https://proceedings.mlr.press/v80/chen18a.html) studies gradient magnitude
balance, and [PCGrad, NeurIPS
2020](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
studies conflicting gradients; neither method is part of the primary matrix.

## Focused follow-up if a mechanistic claim is needed

First finish the prespecified 80 runs and inspect development diagnostics.
For a narrow, prospective follow-up, keep one architecture, dataset, seed,
initialisation, target-task supervision, training steps and selection rule
fixed. Disable one auxiliary task's **loss contribution** while retaining its
modules, then compare target-task test outcomes across the five seeds. This
tests the effect of that auxiliary training signal *within that architecture*.
For soft sharing, use an additional no-coupling control (separate projections
and no L2 link) to distinguish added supervision from the coupling route.
Estimate new GPU time before authorising these extra runs. Do not select a
follow-up because of favourable test outcomes.

An optional, cheaper directional diagnostic is a one-step source-task update
followed by measurement of another task's loss at the same checkpoint. This
is the principle of [Task Affinity Grouping (TAG), NeurIPS
2021](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html).
It would measure a *local, directional optimisation effect*, not prove the
long-run cause of a final test gain. It should be recorded on a fixed probe
across epochs and checked against targeted loss ablations before making a
mechanistic statement. The current project does not claim to implement TAG.

No single architecture can be declared universally superior from one backbone
and one split. A defensible conclusion is conditional: which model gives the
strongest intensity performance at its observed cost, with what emotion and
sentiment trade-offs, on MEISD; and which two-task pattern persists on
BRIGHTER. BRIGHTER does not provide independently annotated emotion presence
or sentiment, so it cannot validate the complete three-task mechanism.
