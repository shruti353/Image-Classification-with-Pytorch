from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mnist-classification",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A CNN-based MNIST digit classification project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mnist-classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.20.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "tensorboard>=2.10.0",
        ],
    },
)