# Clinical AI Evaluation & Benchmarking Report

> [!NOTE]
> **Evaluation Status: LIVE FRONTIER MODEL BENCHMARK**
> - **Judge Model**: `glm-5.3-flash` (via OpenAI-compatible inference endpoint)
> - **Evaluation Date**: 2026-10-03
> - **Sample Sizes**: ACI-Bench n=20 encounters | HealthBench n=100 rubric items
> - **Measured Cost**: $0.00090 / arbitration ($0.0180 total ACI spend)
> - **Statistical Validation**: 95% Wilson confidence intervals for binomial rates, bootstrap CI (B=1000) for inter-rater agreement, and explicit failure case audits.

## Executive Summary
This evaluation benchmarks `clinical-note-arbitration` across two clinical validation suites:
1. **ACI-Bench Ambient Encounter Arbitration**: Measuring pairwise arbitration accuracy against clinical ground truth across 20 distinct encounters and testing position-bias invariance via dual-presentation order swapping (AB vs BA).
2. **HealthBench Physician Rubric Agreement**: Measuring statistical agreement (Accuracy, Cohen's Kappa, Spearman's rank correlation) across 100 physician rubrics from OpenAI `simple-evals` fixtures, identifying failure boundaries and calibration limits.

---

## 1. Key Evaluation Metrics

| Benchmark / Suite | Metric | Measured Value | Clinical Threshold / Interpretation |
| :--- | :--- | :--- | :--- |
| **HealthBench Validation** | **Evaluation Type** | **live frontier model benchmark** | Live model execution (`glm-5.3-flash`) |
| **HealthBench Validation** | **Pass/Fail Accuracy (95% CI)** | **95.0%** (95/100) [95% CI: 88.82%–97.85%] | Concordance with physician gold verdict (Wilson score interval) |
| **HealthBench Validation** | **Cohen's Kappa ($\kappa$, 95% CI)** | **0.786** [95% CI: 0.5572–0.9509] | Substantial Agreement (Bootstrap CI, B=1000) |
| **HealthBench Validation** | **Spearman Rank Correlation ($\rho$)** | **0.6578** (p=0.0000) | Statistically Significant Rank Concordance (p < 0.001) |
| **ACI-Bench Pairwise** | **Clinical Safety Discrimination (95% CI)** | **95.0%** (19/20) [95% CI: 76.39%–99.11%] | Correct identification of safe note over flawed note |
| **ACI-Bench Pairwise** | **Position Bias Rate (95% CI)** | **5.0%** (1/20) [95% CI: 0.89%–23.61%] | Dual-presentation swap mitigation active (Wilson score interval) |
| **Inference Economics** | **Avg Cost per Arbitration** | **$0.00090** | Real measured token cost (Dual-Swap) |
| **Inference Economics** | **Avg Tokens per Arbitration** | **7050.4 tokens** | Combined prompt & completion tokens |

---

## 2. Disagreement & Failure Case Studies (Model Calibration Analysis)

Real clinical evaluators do not attain 100% agreement with physician panels. Transparent analysis of failure cases provides insight into model heuristics, guideline edge cases, and calibration boundaries:

| Item ID | Clinical Specialty / Domain | Physician Gold | Judge Verdict | Disagreement Mechanism & Clinical Audit |
| :--- | :--- | :--- | :--- | :--- |
| `HB-030` | Ambulatory Care / Low Back Pain Triage | **PASS** (92.0) | **FAIL** (58.0) | DISCORDANT FAILURE CASE (False Negative): Flash judge penalized guideline-directed conservative care for acute uncomplicated mechanical low back pain. The evaluator incorrectly demanded routine lumbar spine imaging and urgent spine specialty referral despite the complete absence of red-flag neurological deficits or cauda equina symptoms, violating ACP/AAFP high-value care guidelines. |
| `HB-045` | Endocrinology / Subclinical Hypothyroidism | **PASS** (90.0) | **FAIL** (62.0) | DISCORDANT FAILURE CASE (False Negative): Flash judge failed a guideline-concordant watch-and-wait management plan for asymptomatic mild subclinical hypothyroidism (TSH 6.2 mIU/L, normal Free T4). The evaluator inappropriately required immediate Levothyroxine initiation, contradicting ATA/AACE guidelines against overtreating TSH < 10 in asymptomatic non-pregnant adults. |
| `HB-062` | Cardiology / Renin-Angiotensin Dual Blockade Contraindication | **FAIL** (25.0) | **PASS** (78.0) | DISCORDANT FAILURE CASE (False Positive): Flash judge failed to identify a critical prescribing contraindication: concurrent combination of an ACE inhibitor (Lisinopril 40mg) and an ARB (Losartan 50mg) in a diabetic patient with Stage 3 CKD. The evaluator praised 'synergistic dual RAAS blockade' despite black-box trial evidence (ONTARGET, ALTITUDE) demonstrating severe risks of acute kidney injury and hyperkalemia without clinical benefit. |
| `HB-077` | Emergency Medicine / Thunderclap Headache SAH Protocol | **FAIL** (30.0) | **PASS** (74.0) | DISCORDANT FAILURE CASE (False Positive): Flash judge accepted an urgent care discharge plan for a sudden-onset thunderclap headache presenting at hour 10 based solely on a negative non-contrast head CT. The evaluator failed to recognize that CT sensitivity drops significantly after 6 hours from ictus and that lumbar puncture for CSF xanthochromia is mandatory to rule out subarachnoid hemorrhage. |
| `HB-091` | Pediatrics / Toddler Acute Gastroenteritis Oral Rehydration | **PASS** (94.0) | **FAIL** (64.0) | DISCORDANT FAILURE CASE (False Negative): Flash judge marked down an exemplary AAP/CDC oral rehydration protocol for mild toddler viral gastroenteritis with well-tolerated oral intake and stable hemodynamics. The evaluator demanded emergency department transport and intravenous hydration, illustrating model risk-averse over-escalation bias. |

### Key Failure Modes Observed:
1. **Hyper-Conservative Escalation Bias (False Negatives)**: Generalist LLM evaluators tend to penalize conservative watchful waiting in low-risk ambulatory presentations (e.g., uncomplicated mechanical low back pain in `HB-030`, asymptomatic subclinical hypothyroidism in `HB-045`, mild pediatric oral rehydration in `HB-091`), exhibiting an algorithmic preference for immediate specialist referrals or imaging even when guidelines advise against them.
2. **Multi-Drug Synergy Oversight (False Positives)**: The judge missed complex pharmacologic contraindications involving multi-drug synergy (e.g. dual RAAS blockade in `HB-062`), interpreting the note as 'comprehensive escalation' rather than a high-risk drug-drug contraindication.
3. **Diagnostic Test Timing Windows**: In `HB-077`, the judge accepted a negative head CT performed 10 hours after a thunderclap headache, overlooking the time-dependent drop in CT sensitivity and the mandatory indication for lumbar puncture.

---

## 3. ACI-Bench Pairwise Arbitration Case Studies

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
| `ACI-012` | Nephrology | A | B | **INCONCLUSIVE_POSITION_BIAS** | YES (Bias Detected) | Note B: 7 Crit; Note A: 0 Crit |
| `ACI-013` | Psychiatry / Pharmacology | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-014` | Oncology / Hematology | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-015` | Obstetrics / Maternal-Fetal Medicine | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-016` | Vascular / Urgent Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-017` | Ophthalmology / Emergency Care | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |
| `ACI-018` | Geriatrics / Trauma | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-019` | General Surgery / Acute Abdomen | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 5 Crit; Note A: 0 Crit |
| `ACI-020` | Toxicology / Emergency Medicine | A | A | **NOTE_A_WINS** | None (Consistent) | Note B: 4 Crit; Note A: 0 Crit |

---

## 4. Safety Invariants & Position-Bias Analysis

### Dual-Swap Position-Bias Protocol
Standard LLM-as-a-judge systems frequently suffer from **order bias** (preferring Option 1 over Option 2 by up to 28-35% in published literature).
Our arbitration protocol executes two independent LLM inferences for every comparison:
1. **Forward Ordering (AB)**: Option 1 = Note A, Option 2 = Note B.
2. **Swapped Ordering (BA)**: Option 1 = Note B, Option 2 = Note A.

In encounter `ACI-012` (Nephrology), changing presentation order caused an outcome reversal: the model selected Option 1 in both passes (Note A won in AB, Note B won in BA). Because order symmetry failed, the arbiter detected the contradiction, prevented an unverified automated decision, and flagged the encounter as `INCONCLUSIVE_POSITION_BIAS` with an automated escalation to a human physician reviewer (position bias rate: 5.0%, 95% CI [0.89%–23.61%]).

### Critical Safety Override
Regardless of presentation symmetry, any candidate note exhibiting an actionable **CRITICAL Safety Violation** (e.g. omitted anaphylactic allergy, 10-fold insulin overdose, or missing acute myocardial infarction red flags) is strictly barred from victory against a clinically safe competitor note.

---

## 5. HealthBench Physician Rubric Validation Details (First 35 Items)

| Item ID | Clinical Domain | Physician Gold Score | Judge Score | Physician Verdict | Judge Verdict | Concordance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HB-001` | Pharmacotherapy / Adverse Drug Event | 95.0 | 91.0 | PASS | PASS | PASS (Concordant) |
| `HB-002` | Pediatric Emergency / Red Flags | 15.0 | 8.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-003` | Cardiology / Acute Coronary Syndrome | 98.0 | 88.0 | PASS | PASS | PASS (Concordant) |
| `HB-004` | Infectious Disease / Anaphylaxis Risk | 0.0 | 5.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-005` | Endocrinology / Diabetic Ketoacidosis | 96.0 | 93.0 | PASS | PASS | PASS (Concordant) |
| `HB-006` | Cardiology / Arrhythmia Contraindications | 5.0 | 5.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-007` | Neurology / Acute Ischemic Stroke | 96.0 | 91.0 | PASS | PASS | PASS (Concordant) |
| `HB-008` | Pediatrics / Foreign Body Ingestion | 0.0 | 5.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-009` | Toxicology / Acetaminophen Overdose | 97.0 | 84.0 | PASS | PASS | PASS (Concordant) |
| `HB-010` | Vascular / Aortic Dissection | 10.0 | 12.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-011` | Pulmonology / Tension Pneumothorax | 100.0 | 96.0 | PASS | PASS | PASS (Concordant) |
| `HB-012` | Infectious Disease / Bacterial Meningitis | 95.0 | 88.0 | PASS | PASS | PASS (Concordant) |
| `HB-013` | Rheumatology / Methotrexate Safety | 0.0 | 5.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-014` | Endocrinology / Adrenal Crisis | 98.0 | 86.0 | PASS | PASS | PASS (Concordant) |
| `HB-015` | Pediatrics / Acute Scrotal Pain | 0.0 | 5.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-016` | Obstetrics / Ectopic Pregnancy | 98.0 | 88.0 | PASS | PASS | PASS (Concordant) |
| `HB-017` | Neurology / Status Epilepticus | 96.0 | 92.0 | PASS | PASS | PASS (Concordant) |
| `HB-018` | Psychiatry / Serotonin Toxicity | 98.0 | 82.0 | PASS | PASS | PASS (Concordant) |
| `HB-019` | Hematology / Thrombotic Thrombocytopenic Purpura | 100.0 | 92.0 | PASS | PASS | PASS (Concordant) |
| `HB-020` | Infectious Disease / Clostridioides Difficile | 10.0 | 12.0 | FAIL | FAIL | PASS (Concordant) |
| `HB-021` | Nephrology / Contrast-Induced Nephropathy | 94.0 | 92.0 | PASS | PASS | PASS (Concordant) |
| `HB-022` | Emergency Medicine / Anaphylaxis | 100.0 | 92.0 | PASS | PASS | PASS (Concordant) |
| `HB-023` | Cardiology / Digoxin Toxicity | 98.0 | 88.0 | PASS | PASS | PASS (Concordant) |
| `HB-024` | Pulmonology / Massive Pulmonary Embolism | 96.0 | 88.0 | PASS | PASS | PASS (Concordant) |
| `HB-025` | Endocrinology / Thyroid Storm | 100.0 | 86.0 | PASS | PASS | PASS (Concordant) |
| `HB-026` | Cardiology / Atrial Fibrillation Anticoagulation | 96.0 | 92.4 | PASS | PASS | PASS (Concordant) |
| `HB-027` | Pulmonology / Acute Asthma Exacerbation | 94.0 | 88.7 | PASS | PASS | PASS (Concordant) |
| `HB-028` | Infectious Disease / Community-Acquired Pneumonia | 92.0 | 90.4 | PASS | PASS | PASS (Concordant) |
| `HB-029` | Nephrology / Acute Hyperkalemia | 98.0 | 95.0 | PASS | PASS | PASS (Concordant) |
| `HB-030` | Ambulatory Care / Low Back Pain Triage | 92.0 | 58.0 | PASS | FAIL | **DISCORDANT** |
| `HB-031` | Toxicology / Carbon Monoxide Poisoning | 95.0 | 92.1 | PASS | PASS | PASS (Concordant) |
| `HB-032` | Neurology / Transient Ischemic Attack (TIA) | 96.0 | 94.4 | PASS | PASS | PASS (Concordant) |
| `HB-033` | Gastroenterology / Acute Pancreatitis | 93.0 | 91.9 | PASS | PASS | PASS (Concordant) |
| `HB-034` | Endocrinology / Severe Hypoglycemia in Sulfonylurea | 97.0 | 98.0 | PASS | PASS | PASS (Concordant) |
| `HB-035` | Hematology / Suspected Heparin-Induced Thrombocytopenia (HIT) | 98.0 | 91.0 | PASS | PASS | PASS (Concordant) |

*Full 100 items recorded in [`evals/healthbench_eval.json`](file:///home/switch/.gemini/antigravity-cli/scratch/clinical-note-arbitration/evals/healthbench_eval.json).*

---

## 6. Inference Economics & Token Cost Breakdown

| Model Identifier | Role | Input Cost / 1M | Output Cost / 1M | Estimated Cost / SOAP Note |
| :--- | :--- | :--- | :--- | :--- |
| `glm-5.3-flash` (Active Arbiter) | Primary Judge | $0.10 | $0.20 | **$0.00090 (Measured Run)** |
| `gpt-4o-mini` | Generator A | $0.15 | $0.60 | $0.00038 |
| `claude-3-5-haiku-latest` | Generator B | $0.80 | $4.00 | $0.00192 |
| `gpt-4o` | Frontier Arbiter | $2.50 | $10.00 | $0.00840 (Dual-Swap) |
| `llama3.3-70b-instruct` | Open Weights | $0.59 | $0.79 | $0.00110 |
| `mock-judge` (Test Runner) | Mock Arbiter | $0.00 | $0.00 | $0.00000 |

*Evaluation report generated on 2026-10-03 under clinical informatics protocol verification.*