# A-CUT KonIQ + FLIVE Technical IQA
#### A-CUT Capstone Project — Mobile Technical Image Quality Assessment
> Production technical quality baseline for A-CUT, combining KonIQ and FLIVE image-quality signals for on-device best-shot selection.
## Introduction
This repository documents the technical image quality assessment baseline used in the A-CUT project. Unlike aesthetic models that focus on composition or visual appeal, this model family evaluates technical quality factors such as blur, noise, exposure, and perceptual distortion.
In the current A-CUT production pipeline, the technical score is based on the existing KonIQ + FLIVE path. Other candidates such as TOPIQ, C6, MDIQA, MUSIQ, and student IQA were benchmarked separately, but they are intentionally not included in this repository.
- **Role:** A-CUT production technical quality baseline
- **Task:** No-reference technical image quality assessment
- **Status:** Production baseline
## Model / Method
The technical score combines two mobile-friendly no-reference image quality signals.
- **KonIQ-based model:** captures general perceptual image quality.
- **FLIVE-based model:** captures real-world mobile and in-the-wild technical quality signals.
This repository focuses only on the production KonIQ + FLIVE technical baseline. Research candidates are excluded to keep this repository clean and production-oriented.
## Dataset and Benchmark
Large datasets are not included due to size and license constraints.
The official benchmark used **4,967 image-only test samples**:
| Dataset | Images |
|---|---:|
| KonIQ-10k | 2,015 |
| SPAQ | 1,125 |
| FLIVE image test | 1,827 |
| **Total** | **4,967** |
## Model Weights / Artifacts
Model binaries larger than normal GitHub limits are not included directly. Use [`models/download.md`](models/download.md) for artifact placement and hosting notes.
Recommended hosting for large binaries:
- GitHub Releases
- Git LFS
## Repository Structure
```text
.
├── configs/
├── docs/
├── models/
├── results/
├── scripts/
├── .gitignore
├── LICENSE_PLACEHOLDER.md
├── README.md
└── requirements.txt

Requirements

Install the dependencies listed in requirements.txt.

pip install -r requirements.txt

Requirements are inferred from the copied scripts and reports. Adjust local paths before reproduction.

Quick Start

The benchmark command below is provided as a reproduction reference. Verify paths in docs/reproduction_notes.md￼ before running.

python scripts/run_official_protocol_technical_signal_benchmark.py

Results

Benchmark	Model	SRCC	PLCC	MAE	Severe FP	Evidence Path	Decision
Official 4,967-image protocol	production_existing	0.7168	0.7874	12.00	0	outputs/official_protocol_technical_signal_benchmark_20260604/report.md	Production baseline
Dataset-wise benchmark	KonIQ signal	0.866	-	-	-	outputs/technical_iqa_signal_comparison_20260604/report.md	Production component
Dataset-wise benchmark	FLIVE signal	0.630	-	-	-	outputs/technical_iqa_signal_comparison_20260604/report.md	Production component

Limitations

* This repository documents the production baseline, not every technical IQA experiment.
* Candidate models such as TOPIQ, C6, MDIQA, MUSIQ, and student IQA are intentionally excluded.
* Dataset files and large model binaries are not included.
* Reported values are based on the local A-CUT benchmark protocol and may differ from original paper benchmarks.

Citation

@misc{acut_koniq_flive_technical_iqa_2026,
  title        = {A-CUT KonIQ + FLIVE Technical IQA: Mobile Technical Image Quality Assessment Baseline},
  author       = {Kim, Gwanjung},
  year         = {2026},
  note         = {Capstone project repository}
}

Related Repositories

* A-CUT NIMA Mobile￼
* A-CUT RGNet Mobile￼
* A-CUT Mobile A-LAMP￼

GitHub에서 붙여넣을 때는 README.md 파일 오른쪽 위의 연필 아이콘으로 편집하고, 기존 내용을 전부 지운 뒤 위 내용을 붙여넣으면 된다. 아래쪽 commit message는 예를 들어 이렇게 하면 된다.
```text
docs: polish README formatting
