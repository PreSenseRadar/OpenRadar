import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmwave",
    version="0.9.0",
    author="Edwin Pan, Jingning Tang, Dashiell Kosaka, Arjun Gupta, Ruihao Yao",
    author_email="presenseradar@gmail.com",
    description="A mmWave radar data processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
