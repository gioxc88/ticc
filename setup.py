import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requirements = [req.strip() for req in fh]


setuptools.setup(
    name="ticc",
    version="0.0.1",
    author="Giovambattista Perciaccante",
    author_email="gioxc@hotmail.it",
    description="Multivariate Time Series Clustering Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gioxc88/ticc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=install_requirements,
)
