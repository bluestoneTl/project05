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
find datasets/data_ZZCX_singleHQ -type f > datasets/files.list
# shuffle collected files
shuf datasets/files.list > datasets/files_shuf.list
# pick train_size files in the front as training set
head -n 1300 datasets/files_shuf.list > datasets/files_shuf_train.list
# pick remaining files as validation set
tail -n 1301 datasets/files_shuf.list > datasets/files_shuf_val.list

test:
    find datasets/ZZCX_01_14/test/HQ -type f > datasets/ZZCX_01_14/test/HQ.list
    shuf datasets/ZZCX_01_14/test/HQ.list > datasets/ZZCX_01_14/test/HQ_shuf.list
    find datasets/ZZCX_01_14/test/LQ -type f > datasets/ZZCX_01_14/test/LQ.list
    shuf datasets/ZZCX_01_14/test/LQ.list > datasets/ZZCX_01_14/test/LQ_shuf.list
train:
    find datasets/ZZCX_01_14/train/HQ -type f > datasets/ZZCX_01_14/train/HQ.list
    shuf datasets/ZZCX_01_14/train/HQ.list > datasets/ZZCX_01_14/train/HQ_shuf.list
    find datasets/ZZCX_01_14/train/LQ -type f > datasets/ZZCX_01_14/train/LQ.list
    shuf datasets/ZZCX_01_14/train/LQ.list > datasets/ZZCX_01_14/train/LQ_shuf.list
val:
    find datasets/ZZCX_01_14/val/HQ -type f > datasets/ZZCX_01_14/val/HQ.list
    shuf datasets/ZZCX_01_14/val/HQ.list > datasets/ZZCX_01_14/val/HQ_shuf.list
    find datasets/ZZCX_01_14/val/LQ -type f > datasets/ZZCX_01_14/val/LQ.list
    shuf datasets/ZZCX_01_14/val/LQ.list > datasets/ZZCX_01_14/val/LQ_shuf.list

Stage 1:
    python train_stage1.py --config configs/train/train_stage1.yaml

Stage 2:
    Download pretrained Stable Diffusion v2.1 
    python train_stage2.py --config configs/train/train_stage2.yaml