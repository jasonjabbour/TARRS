from setuptools import setup, find_packages

setup(
    name='safe_ptp',
    version='0.1.0',
    description='TARRS',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jason Jabbour',
    author_email='jasonjabbour@g.harvard.edu',
    url='https://github.com/jasonjabbour/TARRS',
    packages=find_packages(),
    python_requires='>=3.7',
)
