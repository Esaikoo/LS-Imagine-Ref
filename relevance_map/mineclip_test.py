import os
import sys
import traceback
import torch

# ============================================================
# 1. 配置
# ============================================================

# 如果你是在 LS-Imagine 根目录运行：
CKPT_PATH = "../weights/mineclip_attn.pth"

# 如果上面的相对路径找不到，可以改成绝对路径，例如：
# CKPT_PATH = "/root/rivermind-data/mine/projects/LS-Imagine-Ref/weights/mineclip_attn.pth"


def print_ok(msg):
    print(f"\n[OK] {msg}")


def print_fail(msg):
    print(f"\n[FAIL] {msg}")


print("=" * 70)
print("MineCLIP 独立测试")
print("=" * 70)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. 测试 MineCLIP 是否能导入
# ============================================================

try:
    from mineclip import MineCLIP
    print_ok("MineCLIP import 成功")
    print("MineCLIP module:", MineCLIP)

except Exception:
    print_fail("MineCLIP import 失败")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 3. 测试权重文件是否存在
# ============================================================

if not os.path.exists(CKPT_PATH):
    print_fail(f"找不到权重文件: {CKPT_PATH}")
    print("请修改 CKPT_PATH")
    sys.exit(1)

print_ok("找到 MineCLIP 权重")
print("Checkpoint:", os.path.abspath(CKPT_PATH))
print(
    "Checkpoint size:",
    round(os.path.getsize(CKPT_PATH) / 1024 / 1024, 2),
    "MB",
)


# ============================================================
# 4. 创建 MineCLIP
#    这里使用官方 mineclip_attn 参数
# ============================================================

try:
    print("\n正在创建 MineCLIP 模型...")

    model = MineCLIP(
        arch="vit_base_p16_fz.v2.t2",
        hidden_dim=512,
        image_feature_dim=512,
        mlp_adapter_spec="v0-2.t0",
        pool_type="attn.d2.nh8.glusw",
        resolution=[160, 256],
    )

    model = model.to(device)
    model.eval()

    print_ok("MineCLIP 模型创建成功")

except Exception:
    print_fail("MineCLIP 模型创建失败")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 5. 加载 MineCLIP 权重
# ============================================================

try:
    print("\n正在加载权重...")

    model.load_ckpt(
        CKPT_PATH,
        strict=True,
    )

    print_ok("MineCLIP 权重加载成功")

except Exception:
    print_fail("MineCLIP 权重加载失败")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 6. 测试文本编码
#
# 这一部分最重要！
#
# 你刚才的报错就是发生在 encode_text()
# 如果这里报 HuggingFace / tokenizer 网络错误，
# 就说明 MineCLIP 模型本身没坏，问题是 tokenizer。
# ============================================================

prompts = [
    "shear a sheep",
    "find a tree",
    "collect wood",
]

try:
    print("\n" + "=" * 70)
    print("测试 1：MineCLIP 文本编码")
    print("=" * 70)

    print("Prompts:")
    for p in prompts:
        print("  ", p)

    with torch.no_grad():
        text_features = model.encode_text(prompts)

    print_ok("文本编码成功")

    print("text_features.shape =", text_features.shape)
    print("dtype =", text_features.dtype)
    print("device =", text_features.device)

    print("\n第一个文本特征前 10 维：")
    print(text_features[0, :10])

except Exception:
    print_fail("文本编码失败")

    print(
        """
如果下面出现：

huggingface.co
openai/clip-vit-base-patch16
AutoTokenizer.from_pretrained
Network is unreachable

那么就和你刚才 LS-Imagine 的报错完全一致。

也就是说：

MineCLIP 权重加载正常
但是 tokenizer 没有缓存在本地。
"""
    )

    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 7. 测试视频编码
#
# MineCLIP 原版输入：
#
# [B, T, C, H, W]
#
# B = 1
# T = 16
# C = 3
# H = 160
# W = 256
# ============================================================

try:
    print("\n" + "=" * 70)
    print("测试 2：MineCLIP 视频编码")
    print("=" * 70)

    video = torch.randint(
        0,
        256,
        size=(1, 16, 3, 160, 256),
        device=device,
        dtype=torch.uint8,
    )

    print("输入 video.shape =", video.shape)
    print("输入 dtype =", video.dtype)

    with torch.no_grad():

        # ----------------------------------------------------
        # 单帧 / spatial feature
        # ----------------------------------------------------
        image_features = model.forward_image_features(video)

        print("\nimage_features.shape =")
        print(image_features.shape)

        # ----------------------------------------------------
        # temporal aggregation
        # ----------------------------------------------------
        video_features = model.forward_video_features(
            image_features
        )

    print_ok("视频编码成功")

    print("\nvideo_features.shape =", video_features.shape)
    print("video_features.dtype =", video_features.dtype)

    print("\n视频特征前 10 维：")
    print(video_features[0, :10])

except Exception:
    print_fail("视频编码失败")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 8. 测试视频-文本相似度 / reward
# ============================================================

try:
    print("\n" + "=" * 70)
    print("测试 3：MineCLIP 视频-文本匹配")
    print("=" * 70)

    with torch.no_grad():

        logits_per_video, logits_per_text = (
            model.forward_reward_head(
                video_features,
                text_tokens=text_features,
            )
        )

    print_ok("Reward Head 运行成功")

    print("\nlogits_per_video.shape =",
          logits_per_video.shape)

    print("\n视频与三个 prompt 的得分：")

    scores = logits_per_video[0].detach().cpu()

    for prompt, score in zip(prompts, scores):
        print(
            f"{prompt:20s}: {score.item():.6f}"
        )

except Exception:
    print_fail("Reward Head 测试失败")
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# 9. 最终结果
# ============================================================

print("\n")
print("=" * 70)
print("       MineCLIP ALL TESTS PASSED")
print("=" * 70)

print(
    """
✓ MineCLIP 可以 import
✓ MineCLIP 模型可以创建
✓ MineCLIP checkpoint 可以加载
✓ MineCLIP text encoder 正常
✓ MineCLIP vision encoder 正常
✓ MineCLIP temporal encoder 正常
✓ MineCLIP reward head 正常

说明 MineCLIP 基本可以正常使用。
"""
)