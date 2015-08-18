from setuptools import setup

setup(name='anomaly',
      version='0.1.0',
      description='Anomaly detection',
      author='J. Benjamin Cook',
      author_email='jbenjamincook@gmail.com',
      install_requires=[
          'numpy',
          'tdigest'
      ],
      packages=['anomaly'],
      license='Apache License 2.0')
