# Age Prediction using Mixture of Experts (APMoE)

Predicting age using anonymous biometric data with a Mixture of Experts (MoE) model. This project focuses on privacy-preserving age verification without the need for identifiable personal data, integrating advanced Human-Computer Interaction (HCI) implementations to ensure seamless user experience.

## Project Overview

Age verification is critical for protecting minors and ensuring compliance with legal regulations online. Current solutions often require sensitive personal identity documents, raising privacy and security concerns. APMoE offers a novel, privacy-respecting approach by estimating user age from non-identifiable data modalities such as facial images, voice samples, gait, EEG signals, and keystroke dynamics, processed through lightweight deep learning architectures. The project aims to balance accuracy, efficiency, and privacy protection, enabling scalable deployment across platforms and devices.

The system uses ensemble modeling via a mixture of experts, combining predictions from multiple distinct data sources to improve robustness and reliability without compromising user anonymity.

## Features

- Multi-modal age prediction using anonymized biometric and behavioral data
- Lightweight deep learning models like MobileNet and EfficientNet for efficient processing
- Ensemble modeling (Mixture of Experts) for enhanced accuracy
- Evaluation of model fairness, latency, and usability
- Focus on privacy adherence and ethical data handling
- User-friendly verification interface with strong emphasis on data security
- Deployment-ready APIs with containerization support

## Data Sources

Datasets are **not included** in this repository due to licensing restrictions but should be obtained separately from their original sources. The project uses the following datasets:

- Facial Age Dataset (Kaggle): https://www.kaggle.com/datasets/frabbisw/facial-age
- MIMIC Electronic Health Records (EHR): https://mimic.physionet.org/
- OU-ISR Gait Dataset: https://islab.ou.edu/datasets/
- Mozilla Common Voice Speech Dataset: https://commonvoice.mozilla.org/en/datasets
- IKDD Keystroke Dynamics Dataset: https://github.com/MachineLearningVisionRG/IKDD.git
- TUH EEG Corpus: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

## Installation

Instructions for local setup, dependencies, and environment configuration go here. Typical requirements include Python 3.x, TensorFlow/PyTorch, Flask/Docker, and related libraries.

## Usage

Instructions for training models, running predictions, and deploying the verification service.

## Ethical Considerations

This project adheres strictly to data privacy laws and ethical guidelines. No personally identifiable information (PII) is stored, shared, or processed beyond anonymized signals. Predicted ages are used solely for research and prototype development, not for decisions impacting users without additional consent.

## Citation

Please cite this project and datasets appropriately when using or referring to results.

***

## License

The code in this repository is licensed under the Apache 2.0 License for academic and research purposes.

**IMPORTANT:**

- The datasets used in this project are subject to their respective license agreements.
- Redistribution of datasets is **not permitted**; users must obtain datasets directly from their sources.
- This project and its code are for **non-commercial use only**.
- Users **must provide attribution** to the original dataset owners and project creators before using any derived models or results.

By using this repository, you agree to adhere to these licensing and attribution requirements.
