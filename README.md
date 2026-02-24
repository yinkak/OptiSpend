### 📦 Model Storage & GitHub Constraints

The core model output (`mmm_trace.nc`) is a high-dimensional Bayesian trace file approximately **538 MB** in size. 

**Why it is not in this repository:**
* **GitHub File Limits:** GitHub enforces a 100 MB limit per file. 
* **Git Integrity:** Large binary files bloat repository size and slow down cloning for other users.
* **Reproducibility:** In line with ML best practices, this project prioritizes **Code-as-Infrastructure**. Instead of downloading a static model, users are encouraged to generate the trace locally to ensure environment compatibility.

**How to generate the model locally:**
If you wish to perform analysis or run the dashboard, you must generate the trace file on your local machine:
1. Ensure your `venv` is active.
2. Run the training pipeline:
   ```bash
   python3 src/train.py
   ```
3. The script will output the mmm_trace.nc file into the models/ directory, which is ignored by Git.