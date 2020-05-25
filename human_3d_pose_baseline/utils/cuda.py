import torch


def get_device(use_cuda=True):
    """Get CUDA or CPU device used for model training/evaluation.

    Args:
        use_cuda (bool, optional): True if use CUDA.

    Returns:
        (torch.device): Device to use.
    """
    if use_cuda:
        assert torch.cuda.is_available(), "CUDA is not available."

    device = torch.device("cuda" if use_cuda else "cpu")
    return device
