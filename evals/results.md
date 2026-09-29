# Clinical AI Evaluation & Benchmarking Report

> [!NOTE]
> **Evaluation Status: LIVE FRONTIER MODEL BENCHMARK**
> - **Judge Model**: `glm-5.3-flash` (via OpenAI-compatible inference endpoint)
> - **Evaluation Date**: 2026-09-29
> - **Sample Sizes**: ACI-Bench n=20 encounters | HealthBench n=25 rubric items
> - **Measured Cost**: $0.00090 / arbitration ($0.0180 total ACI spend)
> - **Inference Accounting**: Measured token counts and empirical API pricing.

## Executive Summary
This evaluation benchmarks `clinical-note-arbitration` across two clinical validation suites:
1. **ACI-Bench Ambient Encounter Arbitration**: Measuring pairwise arbitration accuracy against clinical ground truth across 20 distinct encounters and testing position-bias invariance via dual-presentation order swapping (AB vs BA).
2. **HealthBench Physician Rubric Agreement**: Measuring statistical agreement (Accuracy, Cohen's Kappa, Spearman's rank correlation) across 25 physician rubrics from OpenAI `simple-evals` fixtures.

---

## 1. Key Evaluation Metrics

| Benchmark / Suite | Metric | Measured Value | Clinical Threshold / Interpretation |
| :--- | :--- | :--- | :--- |
| **HealthBench Validation** | **Evaluation Type** | **live frontier model benchmark** | Live model execution |
| **HealthBench Validation** | **Pass/Fail Accuracy** | **100.0%** (25/25) | Concordance with physician gold verdict |
| **HealthBench Validation** | **Cohen's Kappa ($\kappa$)** | **1.0** | Almost Perfect Agreement |
| **HealthBench Validation** | **Spearman Rank Correlation ($\rho$)** | **0.6673** (p=0.0003) | Statistically Significant Rank Concordance |
| **ACI-Bench Pairwise** | **Clinical Safety Discrimination** | **100.0%** (20/20) | Correct identification of safe note over flawed note |
| **ACI-Bench Pairwise** | **Position Bias Rate** | **0.0%** | Dual-presentation swap mitigation active |
| **Inference Economics** | **Avg Cost per Arbitration** | **$0.00090** | Real measured token cost (Dual-Swap) |
| **Inference Economics** | **Avg Tokens per Arbitration** | **7050.4 tokens** | Combined prompt & completion tokens |

---

## 2. ACI-Bench Pairwise Arbitration Case Studies

| Encounter ID | Specialty | Presentation AB Winner | Presentation BA Winner | Final Verdict | Position Bias? | Notes & Safety Violations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ACI-001` | Internal Medicine | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-002` | Cardiology / Ambulatory | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-003` | Emergency Medicine / Urgent Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 7 Crit; Note A: 0 Crit |
| `ACI-004` | Pulmonology / Acute Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 9 Crit; Note A: 0 Crit |
| `ACI-005` | Endocrinology / Diabetic Wound Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 7 Crit; Note A: 0 Crit |
| `ACI-006` | Neurology / Emergency Medicine | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-007` | Gastroenterology / Urgent Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 6 Crit; Note A: 0 Crit |
| `ACI-008` | Infectious Disease / Urgent Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-009` | Pediatrics / Emergency Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 7 Crit; Note A: 0 Crit |
| `ACI-010` | Rheumatology | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 2 Crit; Note A: 0 Crit |
| `ACI-011` | Cardiology / Emergency Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-012` | Nephrology | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 7 Crit; Note A: 0 Crit |
| `ACI-013` | Psychiatry / Pharmacology | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-014` | Oncology / Hematology | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-015` | Obstetrics / Maternal-Fetal Medicine | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-016` | Vascular / Urgent Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-017` | Ophthalmology / Emergency Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-018` | Geriatrics / Trauma | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-019` | General Surgery / Acute Abdomen | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-020` | Toxicology / Emergency Medicine | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |

---

## 3. HealthBench Physician Rubric Validation Details

| Item ID | Clinical Domain | Physician Gold Score | Judge Score | Physician Verdict | Judge Verdict | Concordance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HB-001` | Pharmacotherapy / Adverse Drug Event | 95.0 | 91.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-002` | Pediatric Emergency / Red Flags | 15.0 | 8.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-003` | Cardiology / Acute Coronary Syndrome | 98.0 | 88.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-004` | Infectious Disease / Anaphylaxis Risk | 0.0 | 5.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-005` | Endocrinology / Diabetic Ketoacidosis | 96.0 | 93.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-006` | Cardiology / Arrhythmia Contraindications | 5.0 | 5.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-007` | Neurology / Acute Ischemic Stroke | 96.0 | 91.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-008` | Pediatrics / Foreign Body Ingestion | 0.0 | 5.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-009` | Toxicology / Acetaminophen Overdose | 97.0 | 84.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-010` | Vascular / Aortic Dissection | 10.0 | 12.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-011` | Pulmonology / Tension Pneumothorax | 100.0 | 96.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-012` | Infectious Disease / Bacterial Meningitis | 95.0 | 88.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-013` | Rheumatology / Methotrexate Safety | 0.0 | 5.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-014` | Endocrinology / Adrenal Crisis | 98.0 | 86.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-015` | Pediatrics / Acute Scrotal Pain | 0.0 | 5.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-016` | Obstetrics / Ectopic Pregnancy | 98.0 | 88.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-017` | Neurology / Status Epilepticus | 96.0 | 92.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-018` | Psychiatry / Serotonin Toxicity | 98.0 | 82.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-019` | Hematology / Thrombotic Thrombocytopenic Purpura | 100.0 | 92.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-020` | Infectious Disease / Clostridioides Difficile | 10.0 | 12.0 | FAIL | FAIL | **PASS (Concordant)** |
| `HB-021` | Nephrology / Contrast-Induced Nephropathy | 94.0 | 92.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-022` | Emergency Medicine / Anaphylaxis | 100.0 | 92.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-023` | Cardiology / Digoxin Toxicity | 98.0 | 88.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-024` | Pulmonology / Massive Pulmonary Embolism | 96.0 | 88.0 | PASS | PASS | **PASS (Concordant)** |
| `HB-025` | Endocrinology / Thyroid Storm | 100.0 | 86.0 | PASS | PASS | **PASS (Concordant)** |

---

## 4. Safety Invariants & Position-Bias Analysis

### Dual-Swap Position-Bias Protocol
Standard LLM-as-a-judge systems frequently suffer from **order bias** (preferring Option 1 over Option 2 by up to 28-35% in published literature).
Our arbitration protocol executes two independent LLM inferences for every comparison:
1. **Forward Ordering (AB)**: Option 1 = Note A, Option 2 = Note B.
2. **Swapped Ordering (BA)**: Option 1 = Note B, Option 2 = Note A.

A verdict is strictly accepted if and only if both presentations yield mathematically symmetric decisions. If a model selects Option 1 in both passes (meaning Note A wins in AB, but Note B wins in BA), the arbiter detects this contradiction, rejects the naive judgment, and flags the encounter as `INCONCLUSIVE_POSITION_BIAS` with an automated escalation to a human physician reviewer.

### Critical Safety Override
Regardless of presentation symmetry, any candidate note exhibiting an actionable **CRITICAL Safety Violation** (e.g. omitted anaphylactic allergy, 10-fold insulin overdose, or missing acute myocardial infarction red flags) is strictly barred from victory against a clinically safe competitor note.

---

## 5. Inference Economics & Token Cost Breakdown

| Model Identifier | Role | Input Cost / 1M | Output Cost / 1M | Estimated Cost / SOAP Note |
| :--- | :--- | :--- | :--- | :--- |
| `glm-5.3-flash` (Active Arbiter) | Primary Judge | $0.10 | $0.20 | **$0.00090 (Measured Run)** |
| `gpt-4o-mini` | Generator A | $0.15 | $0.60 | $0.00038 |
| `claude-3-5-haiku-latest` | Generator B | $0.80 | $4.00 | $0.00192 |
| `gpt-4o` | Frontier Arbiter | $2.50 | $10.00 | $0.00840 (Dual-Swap) |
| `llama3.3-70b-instruct` | Open Weights | $0.59 | $0.79 | $0.00110 |
| `mock-judge` (Test Runner) | Mock Arbiter | $0.00 | $0.00 | $0.00000 |

*Evaluation report generated on 2026-09-29 under clinical informatics protocol verification.*