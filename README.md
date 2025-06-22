# TuneGraph — A Song Recommendation System Based on Graph Collaborative Filtering

*Note: This is a temporary README. A full version will be written once the project is complete, along with a blog post explaining the system in non-technical terms.*

## Introduction

TuneGraph is a song recommendation system being built using **Disentangled Graph Collaborative Filtering (DGCF)**, based on the method proposed by Xiang, Hongye, et al. in their paper [Disentangled Graph Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3397271.3401137?casa_token=QZx0D2_toXQAAAAA:evmz9mGhgze5wj8XNJPaYK9Lb8lNK705SNXNPN0R3D_4BXF0GmSCHonLKPGnCbQbQes3C9gXGXQ6).

This project takes a different route compared to typical machine learning side projects. Rather than jumping on the latest popular model or library, it focuses on implementing a well-grounded but less mainstream research paper — exploring its practical application in music recommendation.

## Setup

1. Create a fresh Python environment (`venv` or `conda` recommended).
2. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Execution

🚧 Project is under development. Execution details will be added as the implementation progresses. As of now the basic Streamlit UI is completed which can be accessed by running

```bash
streamlit run app.py
```

in the `src/` folder. The actual training bench for training and evaluation can be run using

```bash
python runner.py --mode [train,test]
```