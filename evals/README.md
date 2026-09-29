# Evaluation Suite & Clinical Validation Directory

This directory contains the benchmarking datasets, evaluation scripts, and empirical results for `clinical-note-arbitration`.

## Contents

- `results.md`: Comprehensive evaluation results, inter-rater agreement statistics ($\kappa$, $\rho$, accuracy), case studies, and safety error breakdowns (**Live Frontier Model Benchmark** via `glm-5.3-flash`).
- `cost_analysis.md`: Detailed token consumption and financial modeling across commercial and open-source models, with empirical $0.00090/arbitration measured spend.
- `aci_bench_eval.json`: Execution output from pairwise arbitration over ACI-Bench clinical encounter fixtures (n=20 encounters, 141,007 tokens).
- `healthbench_eval.json`: Empirical validation output measuring agreement between the LLM judge and physician-written rubrics from OpenAI `simple-evals` test fixtures (n=25 rubric items, 92,010 tokens).

## Running the Benchmarks

To execute the full evaluation suite against the live model or mock:
```bash
# Live frontier model benchmark (using configured .env credentials)
python scripts/run_evals.py --judge-model glm-5.3-flash --workers 5

# Offline deterministic smoke test (zero API keys required)
python scripts/run_evals.py --judge-model mock-judge
```
Or via the Typer CLI:
```bash
arbiter validate-healthbench --output evals/healthbench_eval.json
arbiter judge --sample-index 0
arbiter report
```
