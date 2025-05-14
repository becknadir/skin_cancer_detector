from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="skin-cancer-detection",
    version="0.1.0",
    author="Author",
    author_email="author@example.com",
    description="A skin cancer detection system using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/skin-cancer-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.9.0",
        "keras>=2.9.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "seaborn>=0.11.0",
    ],
    entry_points={
        "console_scripts": [
            "skin-cancer-detection=src.main:main",
        ],
    },
) 