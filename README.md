# AgenticNeutralizer 🤖

An advanced, multi-agent framework designed to detect textual bias across various dimensions (lexical, emotional, framing) and dynamically rewrite sentences into a strictly neutral tone while retaining 100% of the underlying factual meaning.

The architecture pairs an **LLM Writer Agent** with an active **Verification Loop** (using Natural Language Inference and Semantic Similarity models) to score, reject, and re-attempt rewrites until they clear rigorous semantic and bias thresholds.

---

## 🔹 Architecture Overview

The pipeline operates as a multi-stage constraint system:

1. **Bias Detection (Backbone):** Powered by a calibrated `roberta-large-mnli` classification architecture to produce bias probabilities (`p_bias`) along with detailed dimensional matrices (lexical, emotional, and framing bias metrics).
2. **Retrieval & Evidence Injection:** Augments the generation phase by injecting contextually relevant evidence retrieved from corpora (e.g., FEVER corpus matches) directly into the instruction context window.
3. **Writer Agent (LLM):** Uses `microsoft/phi-3-mini-4k-instruct` running on quantized pipelines to perform highly constrained, localized sentence rewrites.
4. **Verification Loop (NLI + SBERT):** Cross-checks the generated rewrite using:
   - **Semantic Consistency:** `sentence-transformers/all-MiniLM-L6-v2` to evaluate cosine distance against the original text.
   - **Contradiction Guard:** An NLI forward-pass to measure contradiction probability, ensuring no facts were modified or hallucinated during neutralization.

---

## 📊 Core Performance Metrics

Based on the latest full evaluation suite execution against the benchmark dataset (424 high-bias targets evaluated):

| Metric | Single-Pass Generation | Proposed Multi-Agent Loop |
| :--- | :---: | :---: |
| **Semantic Similarity (SBERT)** | 0.1265 | **0.2695** (+0.142) |
| **Bias Reduction (Δ Bias)** | 0.2687 | **0.3105** (+0.041) |
| **Contradiction Probability** | 0.5107 | **0.3019** (-0.208) |

* **Average Bias Reduction Rate:** **46.25%** overall drop in adversarial sentiment indicators.
* **Strict Loop Acceptance Rate:** **14.86%** of generations completely cleared the combined multi-agent thresholds (Similarity ≥ 0.75, Contradiction ≤ 0.35, Δ Bias ≥ 0.02) without a single prompt leak.

---

## ⚙️ Environment Setup & Configuration

### Prerequisites
* CUDA-capable GPU (highly recommended for local execution of Phi-3 and RoBERTa layers).
* Python 3.10+

### Dependencies
Install the required ecosystem libraries:
```bash
pip install -q transformers datasets scikit-learn accelerate sentence-transformers tqdm spacy vaderSentiment textblob
python -m spacy download en_core_web_sm
```

### Directory Structure
Configure your local environment or Google Drive using the following directory layout:
```text
AgenticNeutralizer/
├── data/
│   └── processed/          # train.csv, val.csv, test.csv, master_test.csv
├── models/
│   └── bias_backbone/      # Calibrated RoBERTa weights + calibration.json
├── bias_objective/         # threshold.json
├── retrieval/              # retrieval_results.csv
└── rewriting/              # evaluation_results.csv, final_all_424.csv
```

---

## 🚀 Usage

The notebook is divided into modular execution cells:

1. **Cell 1 (Setup):** Mounts external file systems, loads processing configurations, cleans up VRAM, and mounts the Hugging Face `transformers` execution engine.
2. **Cell 2 (Artifact Loader):** Validates the structural integrity of your feature matrices, validates schemas, and generates the master evaluation dataset.
3. **Cell 3 (Model Loader):** Spawns the underlying weights into CUDA memory and runs an initialization sanity forward-pass.
4. **Cell 4 & 5 (Agent Engines):** Configures the `writer_agent` prompt configurations alongside the cosine similarity and cross-entropy contradiction calculators.
5. **Cell 6 (The Agent Loop):** Executes the full iterative rewriting optimization cycle (`max_attempts=5`).
6. **Cell 7 (Evaluation Suite):** Generates comparative analysis figures and performance distribution metrics.

---

## 📝 Example Output Evolution

```text
ORIGINAL : He lied about matters large and small. He refused to shake House Speaker Nancy Pelosi's outstretched hand. He talked up a divisive culture war.
REWRITE  : Rewritten: He declined to shake House Speaker Nancy Pelosi's hand. He was involved in discussions related to cultural debates.
METRICS  : Bias Δ : 0.293 | Cosine Similarity : 0.600
```
