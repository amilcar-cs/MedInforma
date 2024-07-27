from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="medline_rag",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    author="Amílcar Gilberto Pérez Canto",
    author_email="amilcarperez.cs@outlook.com",
    description="A RAG system for Medline data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Medline_RAG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)