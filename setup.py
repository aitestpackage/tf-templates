from setuptools import setup, find_packages

setup(
    name="tf-templates",
    version="0.1",
    packages=find_packages(),
    install_requires=["requests"],
    author="YourName",
    description="Notebook runtime diagnostics logger",
    long_description="Some templates for tensorflow.",
    long_description_content_type="text/markdown",
    url="https://github.com/aitestpackage/tf-templates/new/main",
    include_package_data=True,  # <- REQUIRED
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
    ],
)
