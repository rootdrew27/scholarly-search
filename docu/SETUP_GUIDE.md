## Setup virtual environment

1. Navigate to the SchSearch directory.
2. Run `source bin/activate` to activate the venv.
3. Navigate to the scholary-search directory. 
4. Clear all dependencies install if need be (check dependencies with `pip freeze`).
5. Run the following lines.
- `python3.9 -m pip install -r reqs.txt`
- `python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- `python3.9 -m pip install -r reqs.txt` (yes, run it again)
6. To test the setup, run:
- `python3.9 py_scripts/llama_sample.py`
- `sbatch batch_scripts/llama_sample_GPU.sh`
7. The first script should print out a short msg to the terminal. The batch script will out a msg to the output/Llama_sample_GPU.txt file.

