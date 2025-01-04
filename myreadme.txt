python  inference.py 
--task denoise 
--upscale 1 
--version v2.1 
--captioner llava 
--cfg_scale 8 
--noise_aug 0 
--input inputs/demo/bid 
--output results/v2.1_demo_bid

单卡训练：

数据集处理
# collect all iamge files in img_dir
find datasets/data_ZZCX_singleHQ -type f > files.list
# shuffle collected files
shuf files.list > files_shuf.list
# pick train_size files in the front as training set
head -n 1300 files_shuf.list > files_shuf_train.list
# pick remaining files as validation set
tail -n 1301 files_shuf.list > files_shuf_val.list

Stage 1:
    python train_stage1.py --config configs/train/train_stage1.yaml

Stage 2:
    Download pretrained Stable Diffusion v2.1 
    python train_stage2.py --config configs/train/train_stage2.yaml