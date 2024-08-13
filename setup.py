from setuptools import setup, find_packages

if __name__ == "__main__":
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()

    setup(
        name="torchlogic",
        packages=find_packages('torchlogic'),
        package_dir={'': 'torchlogic'},
        version="0.0.3-beta",
        description="A PyTorch framework for rapidly developing Neural Reasoning Networks.",
        classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
        authors="Anonymous",
        python_requires=">=3.6",
        install_requires=requirements,
    )