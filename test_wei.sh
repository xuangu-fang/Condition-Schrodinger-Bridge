#conda env remove -n sb-fbsde
# our cuda is 11 instead of 10 in the original code (pretty slow in our settings)
#conda env create --file requirements_wei.yaml python=3

#conda activate sb-fbsde

#python main.py --problem-name moon-to-spiral --forward-net toy --backward-net toy  --dir ./output --log-tb
python main.py --problem-name Scurve --forward-net toy --backward-net toy  --dir ./output #--log-tb
#python main.py --problem-name cifar10 --forward-net ncsnpp --backward-net Unet --num-FID-sample 10000 --dir ./output --log-tb 
#python main.py --problem-name mnist --forward-net Unet --backward-net Unet --dir ./output --log-tb
