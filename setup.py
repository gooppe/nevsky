from setuptools import find_packages, setup

setup(
    name="nevsky",
    version="0.0.1",
    packages=find_packages(),
    description="Simple and lightweighted translation bot",
    python_requires=">=3.6.0",
    install_requires=[
        "torch==1.4.0",
        "youtokentome>=1.0.6",
        "tqdm",
        "numpy",
        "pyTelegramBotAPI",
        "pyyaml",
    ],
    entry_points={"console_scripts": ["nevsky = nevsky.cli:main"]},
)
