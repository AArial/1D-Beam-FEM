# 1D Beam FEM Simulator — TgN Virtual Lab

Interactive 1D Euler-Bernoulli Beam Finite Element Analysis web app built with Streamlit.

## Features
- 7 beam types: Simply Supported, Cantilever, Fixed-Fixed, Propped Cantilever, Elastic Support, Overhang, Custom
- 4 cross-section profiles: Rectangle, Circle, I-Beam, Hollow Rectangle (auto-calculates I and W)
- Editable node table: prescribed displacements, rotations, forces, moments, translational and rotational springs
- Live model preview with support symbols and load arrows
- FEM solver: Hermitian cubic shape functions, global stiffness assembly, Gaussian elimination
- Results: deformed shape, bending moment diagram, bending stress diagram, result tables
- Global stiffness matrix K and vectors F, u display
- Download results as PNG and CSV

## Theory
Euler-Bernoulli beam with Hermitian cubic shape functions.
Element stiffness matrix assembled from 4×4 local matrices.
Boundary conditions applied by exact elimination method.

## Deploy to Streamlit Community Cloud

1. Fork or push this folder to a **public GitHub repository**
2. Go to https://share.streamlit.io/new
3. Connect your GitHub account
4. Select your repository → branch → `app.py`
5. Click **Deploy**

The app runs on `requirements.txt` — no additional configuration needed.

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
TgN — Naypyitaw State Polytechnic University, Myanmar  
Blog: https://finiteelementsimulationsbytgn.blogspot.com  

