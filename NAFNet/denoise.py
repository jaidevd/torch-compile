from basicsr.models.archs.NAFNet_arch import NAFNet
import torch
import cv2
import matplotlib.pyplot as plt


def load_image(path, device="cpu"):
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0
    return x.unsqueeze(0).to(device)


def load_model(path="experiments/pretrained_models/NAFNet-SIDD-width64.pth"):
    # This is the original torch.nn.Module
    net_g = NAFNet(width=64, enc_blk_nums=[2, 2, 4, 8],
                   middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
    checkpoint = torch.load(path)
    net_g.load_state_dict(checkpoint['params'], strict=True)
    return net_g.eval()


if __name__ == "__main__":
    x = load_image('demo/noisy.png')
    model = load_model()
    print('Tracing...')
    traced = torch.jit.trace(model, (x,))
    with torch.no_grad():
        y = traced(x)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(x[0].permute(1, 2, 0))
    ax[1].imshow(y[0].permute(1, 2, 0))
    [a.axis("off") for a in ax]
    plt.show()
