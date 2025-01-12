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
    wget https://hf-mirror.com/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
    python train_stage2.py --config configs/train/train_stage2.yaml

测试命令：
python -u inference.py --task denoise --upscale 1 --version v2 --sampler spaced --steps 50 --captioner none --pos_prompt '' --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' --cfg_scale 4.0 --input datasets/ZZCX_01_14/test/LQ --output results/1_7_1 --device cuda --precision fp32

自定义模型的测试命令：
python -u inference.py \
--upscale 1 \
--version custom \
--train_cfg configs/train/train_stage2.yaml \
--ckpt experiment2/stage2/checkpoints/0030000.pt \
--captioner llava \
--cfg_scale 8 \
--noise_aug 0 \
--input datasets/ZZCX_01_14/test/LQ \
--output results/1.12/custom \
--precision fp16

推理实验  
1_7     都是用自定义模型测试,  --captioner none
1_7_2   都是用自定义模型测试   --captioner llava
1_7_3   v1的denoise测试  
1_7_4   v2的denoise测试    更改了bid_loop.py的v2加载模型为 swinir

注意，在pretrained_models.py中更改了模型的加载路径，在common.py中更改了加载方式
去噪案例命令：
python -u inference.py \
--task denoise \
--upscale 1 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner llava \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4.0 \
--input datasets/ZZCX_01_14/test/LQ \
--output results/1.12/denoise \
--device cuda \
--precision fp16

【1.12推理实验】    在results/1.12 下
custom             自定义模型测试 --precision fp16   全黑
custom_1           自定义模型测试 --precision fp32   出现图片，但奇怪质量
denoise            去噪案例命令  