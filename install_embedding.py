#Huggingface login
# If you don't have uv yet (one-liner installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make sure ~/.local/bin is on PATH (add to ~/.bashrc if needed)
export PATH="$HOME/.local/bin:$PATH"

# Check it works
uv --version


#Create a project folder + virtual env:
mkdir embeddings && cd embeddings

# Create a venv called .venv
uv venv

# Activate it
source .venv/bin/activate

#2.1 Install PyTorch with CUDA
uv pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu124

#2.2 Install the rest
uv pip install \
    "transformers>=4.44" \
    "datasets>=2.20" \
    huggingface_hub

#huggingface login with tokens from notes
huggingface-cli login

uv pip install hf_transfer
uv pip install scikit-learn

#------------------New Procedure----------------------
# 1. Clone your repo (if not already there)
git clone https://github.com/xingweiGan/Similar_Group.git
cd Similar_Group

# 2. Run the setup once
bash scripts/setup_env.sh

# 3. !!!!Activate venv
source embeddings/.venv/bin/activate

# 4. Login to Hugging Face (first time on this machine)
huggingface-cli login