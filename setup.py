from setuptools import find_packages, setup

setup(
    name="ssnb",
    packages=[package for package in find_packages() if package.startswith("ssnb")],
    url="https://github.com/PaulTiberiu/Reinforcement-Learning",
    version="0.0.1",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    author="EL MOUAHID Soufiane, IORDACHE Paul-Tiberiu, LUONG Ethan, MEKHAIL Paul",
    description="Scalability for Swimmer with N Bodies",
    long_description=open("README.md").read(),
)