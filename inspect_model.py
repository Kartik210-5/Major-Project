import torch

MODEL_PATH = "grayscale_lung_wgan_epoch_75.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

print("Loaded object type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys in checkpoint:")
    for key in checkpoint.keys():
        print(key)

    # Inspect model parameter names and dimensions
    state_dict = checkpoint

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    if isinstance(state_dict, dict):
        print("\nParameter names and shapes:")
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                print(name, tuple(value.shape))
else:
    print(checkpoint)
