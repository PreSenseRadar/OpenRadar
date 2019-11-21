import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def versionDev():
    from setuptools_scm.version import get_local_dirty_tag
    def clean_scheme(version):
        return get_local_dirty_tag(version) if version.dirty else ''

    return {'local_scheme': clean_scheme}

setuptools.setup(
    name="openradar",
    version="1.0.0",
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
    use_scm_version = versionDev,
    setup_requires=['setuptools_scm'],
)
