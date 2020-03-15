from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='robics',
    packages=['robics'],
    version='0.11',
    license='MIT',
    description='Automatic detection of robust parametrizations for LDA and NMF. Compatible with scikit-learn and gensim.',
    author='Christoph Kralj',
    author_email='christoph.kralj@gmail.com',
    url='https://github.com/Christoph/robics',
    download_url='https://github.com/Christoph/robics/archive/v_011.tar.gz',
    keywords=['nlp', 'Topic Model', 'sklearn', 'gensim', 'topic-modeling'],
    install_requires=[            # I get to this in a second
        'numpy',
        'scipy',
        'sobol_seq',
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
