from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pyside2",
        "matplotlib",
        "scikit-learn",
        "openpyxl",
        "scikit-image",
        "pandas",
        "pyopengl",
        "pyqtgraph",
        "pyinstaller"
    ],
)