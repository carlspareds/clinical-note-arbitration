# Dataset Provenance, Ethics & Healthcare Compliance (DATA.md)

## 1. Overview and Core Ethics Policy

This repository implements rigorous AI benchmarking for ambient clinical scribe systems and medical large language models. To safeguard patient confidentiality and comply with international medical data privacy standards, this project strictly adheres to the following principles:

1. **NO REAL PATIENT DATA / ZERO PROTECTED HEALTH INFORMATION (PHI)**:
   Under no circumstances is real-world identifiable patient data ingested, processed, or committed to this repository. All dialogues and notes used in test suites and sample fixtures are fully synthetic or de-identified public research benchmarks.
2. **NO CREDENTIALED OR RESTRICTED-ACCESS DATASETS**:
   This project strictly prohibits the use of PhysioNet credentialed datasets, including **MIMIC-III, MIMIC-IV, eICU**, or National Center for Cognitive Informatics in Health (n2c2 / i2b2) data that require individual Data Use Agreements (DUAs) or institutional CITI training certifications.
3. **NO RAW DATA COMMITTED TO GIT**:
   Raw datasets are isolated in `data/raw/` (configured in `.gitignore`). Only tiny, sanitized sample fixtures (`data/samples/`) are tracked in the repository for unit testing and offline continuous integration.

---

## 2. Public Dataset Provenance & Licenses

| Dataset | Source / Authors | Primary Use Case | Access & License | Citations / Repository |
| :--- | :--- | :--- | :--- | :--- |
| **ACI-Bench** | Yim et al. (Stanford / Microsoft Research) | Doctor-patient dialogue to full SOAP clinical notes with physician reference notes | Open Research / Permissive | [ACI-Bench Repository](https://github.com/clinical-nlp/aci-bench) |
| **MTS-Dialog** | Abacha et al. (National Library of Medicine / NIH) | Doctor-patient dialogue to clinical section summaries | Creative Commons Attribution 4.0 (CC BY 4.0) | [MTS-Dialog Repository](https://github.com/abachaa/MTS-Dialog) |
| **HealthBench** | OpenAI `simple-evals` clinical evaluation suite | Physician-written gold-standard rubrics for clinical decision support | MIT License | [OpenAI simple-evals](https://github.com/openai/simple-evals) |

### Detailed Dataset Descriptions

### A. ACI-Bench (Ambient Clinical Intelligence Benchmark)
- **Provenance**: Developed by clinical NLP researchers to evaluate ambient scribe technologies translating spoken encounters into structured clinical documentation.
- **Content**: Multi-turn clinical dialogues across multiple specialties (Internal Medicine, Cardiology, Urgent Care, Family Practice) paired with expert physician-authored SOAP notes.
- **Licensing**: Open research access. Raw transcripts are dynamically verified via `scripts/download_datasets.py`.

### B. MTS-Dialog (Medical Dialogue Summarization)
- **Provenance**: Created by the US National Library of Medicine (NLM/NIH) and presented at BioNLP / ACL shared tasks.
- **Content**: 1,700 de-identified and simulated medical conversations paired with concise clinical summaries organized by SOAP sections.
- **Licensing**: CC BY 4.0.

### C. HealthBench (OpenAI simple-evals)
- **Provenance**: Curated by licensed physicians for OpenAI's `simple-evals` framework to evaluate medical safety, diagnostic triage, adverse drug reactions, and urgent red-flag recognition.
- **Content**: Hundreds of clinical scenarios with granular physician evaluation rubrics, pass/fail thresholds, and error severities.
- **Licensing**: MIT License.

---

## 3. Regulatory Alignment

### A. HIPAA Privacy Rule (United States)
- All sample dialogues conform to the Safe Harbor de-identification method under 45 CFR § 164.514(b)(2), with complete omission of all 18 HIPAA identifiers (names, geographic subdivisions, dates, phone numbers, MRNs, biometric identifiers, etc.).

### B. EU AI Act (Regulation (EU) 2024/1689)
- Clinical documentation software used in patient diagnosis or therapy planning is classified as **High-Risk AI System** (Annex III).
- Article 10 (Data and data governance): Datasets must be relevant, representative, and free of discriminatory bias.
- Article 14 (Human oversight): The LLM arbiter is designed as an automated audit companion; all flagged inconsistencies (`INCONCLUSIVE_POSITION_BIAS`) trigger mandatory physician review.
- Article 72 (Post-market monitoring): The continuous arbitration pipeline enables ongoing surveillance of deployed AI scribes in hospital production environments.

### C. EU Medical Device Regulation (MDR 2017/745)
- Under MDR Rule 11 (Annex VIII), software intended to provide information used to take decisions with diagnosis or therapeutic purposes falls under Class IIa or IIb medical device software. This repository provides post-market validation tools for validating LLM clinical outputs against physician gold standards.
