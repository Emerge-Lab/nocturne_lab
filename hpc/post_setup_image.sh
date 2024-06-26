# Script for installing Nocturne (lab version) and dependencies in Singularity image
# ---
# Note: Script must be run from the Nocturne LAB repository (i.e. /scratch/$USER/nocturne_lab/)
# and only once after running 'bash ./hpc/launch_image.sh' for the first time.

# Install requirements before installing Nocturne
echo 'Installing installation dependencies: pip, poetry, and linking git submodules...'
pip install -U pip poetry
git submodule sync
git submodule update --init --recursive

# Install Nocturne
echo 'Installing Nocturne using poetry...'
poetry build  # Note: Build first to see build outputs. If build gets stuck at 38%, increase RAM.
rm setup.py
poetry install --all-extras
echo "Successfully installed Nocturne."
echo "To use Nocturne, please close the current Singularity image by running \
  'exit' and afterwards relaunch the image by running 'bash ./hpc/launch_image.sh'"
