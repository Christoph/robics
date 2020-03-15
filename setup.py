from distutils.core import setup
setup(
    name='robics',
    packages=['robics'],
    version='0.1',
    license='MIT',
    description='Automatic detection of robust parametrizations for LDA and NMF. Compatible with scikit-learn and gensim.',
    author='Christoph Kralj',
    author_email='christoph.kralj@gmail.com',
    url='https://github.com/Christoph/robics',
    download_url='https://github.com/Christoph/robics/archive/v_01.tar.gz',
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
)
