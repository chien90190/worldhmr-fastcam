#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X neutral model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './data/body_models/smplx.zip' --no-check-certificate --continue
unzip data/body_models/smplx.zip -d data/body_models/smplx
mv data/body_models/smplx/models/smplx/SMPLX_NEUTRAL.npz data/body_models/smplx/
rm -rf data/body_models/smplx/models
rm -rf data/body_models/smplx.zip


# SMPL neutral model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models/smpl
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './data/body_models/smpl/smpl.zip' --no-check-certificate --continue
unzip data/body_models/smpl/smpl.zip -d data/body_models/smpl/smpl
mv data/body_models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl data/body_models/smpl/SMPL_NEUTRAL.pkl
rm -rf data/body_models/smpl/smpl
rm -rf data/body_models/smpl/smpl.zip

# SMPL-X to SMPL conversion
gdown --folder -O ./data/ https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7?usp=share_link
rm -f data/body_models/J_regressor_h36m.npy data/body_models/smpl_mean_params.npz
