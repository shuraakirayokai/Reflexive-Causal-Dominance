# Reflexive Causal Dominance in Creator-Created Systems

Regulatory policy and simulation code registry associated with the technical manuscript intended for publication.

## Abstract
This project introduces an operational definition of creation in reflexive dynamical systems based on causal dominance over future state trajectories rather than historical origin. Dominance is quantified via interventional effects on a persistence metric $\tau$, comparing the influence of an external controller policy $\pi_D$ against a system's internal self-regulatory policy $\pi_S$. 

Using simulation-based $do()$ interventions, we estimate $\Delta_D$ and $\Delta_S$ and evaluate the structural ratio:
$$\text{Dominance}(D \rightarrow S) = \frac{|\Delta_D|}{|\Delta_S| + \epsilon}$$

Experimental results confirm that internal self-regulation dominates external control ($\text{Dominance} = 0.430 < 1.0$), proving the transience of the creator role as autonomous regulatory capacity emerges.

## Repository Contents
* `reflexive_causal_dominance.py`: Core production-ready Python EDE solver with embedded $do()$-calculus intervention engines.
* `reflexive_dominance_plot.png`: Multi-panel publication-grade 300 DPI visualization layout.
* `Reflexive_Causal_Dominance_Manuscript.txt`: Compiled technical manuscript.
* `Reflexive_Causal_Dominance_Manuscript.docx`: Microsoft Word editable layout for journal submission.

## Requirements & Execution
The simulation utilizes standard scientific Python stacks. To install dependencies and run the production pipeline:
```bash
pip install numpy matplotlib
python reflexive_causal_dominance.py
```

## Citation
Martin, D. B. (2026). Reflexive Causal Dominance and Persistent Cooperative Dynamics in Creator-Created Systems. Zenodo Digital Registry.
