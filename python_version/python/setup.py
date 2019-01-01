import setuptools

with open("readme.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ESN",
    version="0.0.1",
    author="Zeeshan Malik",
    author_email="doctordotmalik@gmail.com",
    description="Multiple Layer ESN",
    url="https://github.com/Xeeshanmalik/deep_ml_esn",
    packages=setuptools.find_packages()
)
