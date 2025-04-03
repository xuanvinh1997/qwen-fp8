from setuptools import setup, find_packages

setup(
    name="qwen_25_moe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "numpy>=1.18.0",
        "pandas>=1.0.0",
    ],
    author="Vinh Pham",
    author_email="phamxuanvinh023@gmail.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/xuanvinh1997/distill-model-on-edge",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)