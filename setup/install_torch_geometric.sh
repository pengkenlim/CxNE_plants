#create virutal 
#pip install virtualenv
#virtualenv -p python .CxNE_env
#source ./.CxNE_env/bin/activate
#pip install --upgrade pip
#pip install -r ./setup/requirements_wo_torch_dependencies.txt
#To install pytorch and geometric
##  uncomment one of the two options


#install gdown to download data
pip install gdown

#OPTION 1
##  pip, python, build = 2.4.1, Linux, CUDA= 12.4
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

#OPTION 2
##  pip, python, build = 2.4.1, Linux, CPU
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip install torch_geometric
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html