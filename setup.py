import os
import unittest
from setuptools import setup, find_packages


def discover_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, mode='r', encoding='utf-8') as f:
        return f.read()


setup(
    name='bert_ner',
    description='BERT for NER using catalyst',
    author='Dmitry Kryuchkov',
    author_email='xelibrion@gmail.com',
    url='https://github.com/xelibrion/bert-ner-catalyst-starter',
    packages=find_packages(),
    setup_requires=[
        'pytest-runner',
    ],
    install_requires=read('requirements.txt').splitlines(),
    extras_require={
        'dev': [
            'pytest',
            'pytest-watch@git+https://github.com/xelibrion/pytest-watch.git@0b9eb018ad1b9a3f9dbd550559cd981b98f512d5',
            'yapf',
        ],
    },
    tests_require=[
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'run-train = bert_ner.train:main',
            'run-inference = bert_ner.inference:main',
        ],
    },
    test_suite='setup.discover_tests',
    zip_safe=True,
)
