from setuptools import setup, find_packages

setup(
    name="bsd",
    version="1.0",
    description="Bayesian Stochastic Depth",
    author="Leon Yao",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "fvcore",
        "tqdm",
    ],
)
