import os
from setuptools import setup
from setuptools import find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "colorbot-trainer",
    version = "0.0.1",
    author = "Andrew Ortman",
    author_email = "andrew@cleverpebble.com",
    description = ("An attempt to create an ML model that learns color names"),
    license = "MIT",
    keywords = "tensorflow color recurrent lstm",
    url = "http://colorbot.cleverpebble.com",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    long_description=read('./README.md'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires = [
        "tensorflow-gpu==1.0.1",
    ],
)
