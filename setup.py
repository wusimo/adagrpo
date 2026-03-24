from setuptools import setup, find_packages

setup(
    name="adagrpo",
    version="0.1.0",
    description="Adaptive Group Relative Policy Optimization for Diffusion-Based VLA Models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "diffusers",
        "transformers",
        "hydra-core",
        "omegaconf",
        "wandb",
        "gymnasium",
        "einops",
        "tqdm",
        "numpy",
    ],
    extras_require={
        "libero": ["libero"],
        "robomimic": ["robomimic"],
        "maniskill": ["mani_skill"],
        "all": ["libero", "robomimic", "mani_skill"],
    },
)
