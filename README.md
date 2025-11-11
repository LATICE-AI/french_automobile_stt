# Latice.AI – French Automotive STT Benchmark

This repository hosts the end-to-end benchmark we ran on the **French Automobile STT** dataset to compare Latice.AI’s automotive speech-to-text engine against leading general-purpose STT providers.

---

## 🎯 Objective
- Demonstrate Latice.AI’s accuracy advantage on domain data
- Measure robustness on raw + augmented utterances (437 total)

---

## 🔬 Methodology
- **Dataset**: 437 French automotive utterances (raw + augmented). `Audio/`, `dataset.json`, and `result.csv` are perfectly aligned.
- **Models Evaluated (9)**: Deepgram Nova 3, ElevenLabs, Fal, Gladia, Google latest_long, Groq Whisper Large V3, Groq Whisper Large V3 Turbo, **Latice.AI**, OpenAI GPT-4o Transcribe.
- **Metrics**: Word Error Rate (WER), perfect vs failed utterances, error-type distributions, qualitative examples.
- **Scripts**: `launch_test.py` for bulk transcription, `wer.py` for normalized WER

---

## 📊 Results

### Average WER (lower is better)
| STT Service | Average WER | Delta vs Latice |
|-------------|-------------|-----------------|
| **Latice.AI** | **0.199** | **Baseline** |
| Fal | 0.372 | -86.9% |
| Groq Whisper Large V3 | 0.372 | -86.9% |
| Groq Whisper Large V3 Turbo | 0.386 | -94.0% |
| ElevenLabs | 0.406 | -104.0% |
| OpenAI GPT-4o Transcribe | 0.408 | -105.0% |
| Gladia | 0.418 | -109.9% |
| Deepgram Nova 3 | 0.440 | -121.1% |
| Google latest_long | 0.507 | -155.3% |

### Perfect vs Failed Transcriptions (437 audios)
| STT Service | Perfect (WER=0) | Failed (WER=1) |
|-------------|-----------------|----------------|
| **Latice.AI** | **254** | **23** |
| Fal | 151 | 41 |
| Groq Whisper Large V3 | 150 | 41 |
| Groq Whisper Large V3 Turbo | 152 | 60 |
| ElevenLabs | 141 | 59 |
| OpenAI GPT-4o Transcribe | 135 | 74 |
| Gladia | 108 | 91 |
| Deepgram Nova 3 | 126 | 87 |
| Google latest_long | 102 | 94 |

### Dataset Alignment Summary
- `Audio/` wav files: **437**
- `dataset.json` entries: **437**
- `result.csv` rows (unique samples): **437**
- `wer_results.txt` “expected” blocks: **426** (identical raw/aug texts are merged in that export)

---

## 🏆 Key Findings
- **Accuracy Lead**: Latice.AI delivers 19.9% WER vs 37–51% for other providers.
- **Robustness**: 254 perfect transcriptions, only 23 complete failures.
- **Coverage**: Six core automotive categories, with 51 to 178 samples per category.
- **Ready-to-Ship Workflow**: CSV + visuals + Word/PDF report generated automatically (`english_analysis_generator/`).

---

## 📈 Business Impact
- **Higher Accuracy** → fewer QA cycles for downstream teams.
- **Reliable Operations** → consistent output on VINs, plates, jargon, customer intents.
- **Process Efficiency** → automated benchmarking pipeline for client-facing deliverables.

---

## 📋 Repository Structure
```
├── Audio/                       # 437 automotive audio clips
├── dataset.json                 # Ground-truth transcriptions (aligned)
├── result.csv                   # STT model outputs
├── wer_results.txt              # Text-based WER export (grouped by prompt)
├── launch_test.py               # Batch STT evaluation pipeline
├── wer.py                       # Normalized WER computation
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started
1. Run bulk transcriptions: `python launch_test.py`
2. Compute normalized WER & export: `python wer.py`

---

## 📊 Conclusion
Latice.AI’s automotive STT outperforms every general-purpose engine we tested. With 2× better accuracy and dramatically fewer failure cases, the solution is tuned for real-world dealership and customer-support use cases.

> **Latice.AI delivers the best-in-class accuracy, robustness and reporting workflow for French automotive speech-to-text.**
