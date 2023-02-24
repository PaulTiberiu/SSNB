from setuptools import find_packages, setup

setup(
    name="ssnb",
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith("ssnb")],
    install_requires=open("requirements.txt", "r").read().splitlines(),
    url="https://github.com/PaulTiberiu/SSNB",
    author="EL MOUAHID Soufiane, IORDACHE Paul-Tiberiu, LUONG Ethan, MEKHAIL Paul",
    description="Scalability for Swimmer with N Bodies",
    long_description=open("README.md").read(),
)
