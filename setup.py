import setuptools

with open('README.md') as fh:
    long_description = fh.read()

with open('requirements.txt') as fp:
    install_requires = fp.read()

setuptools.setup(
    name='LabelAugmented',
    version='0.0.1',
    description='Using memory from previous samples',
    long_description=long_description,
    long_description_conttype='text/markdown',
    packages=setuptools.find_packages(exclude=['notebooks']),
    install_requires=install_requires,
    zip_safe=False,
)
