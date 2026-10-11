
import torch
import matplotlib.pyplot as plt
from model import Generator

CHECKPOINT_PATH = "grayscale_lung_wgan_epoch_75.pth"
OUTPUT_PATH = "generated_image.png"
LATENT_DIM = 100

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

generator = Generator().to(device)

generator.load_state_dict(checkpoint["generator_state_dict"])
generator.eval()

with torch.no_grad():
    noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
    image = generator(noise)

image = image.squeeze(0).squeeze(0).cpu()
image = (image + 1) / 2

plt.imshow(image, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0)
plt.close()

print(f"Generated image saved to {OUTPUT_PATH}")
print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
print(f"Device used: {device}")
