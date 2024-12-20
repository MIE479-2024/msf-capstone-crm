from setuptools import setup, find_packages

setup(
    name="loan_classifier",
    version="0.1.0",
    description="Loan deliquency prediction package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'utils': ['models/*.pkl'],  # Include pickle files from utils/models
    },
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "optbinning"
    ],
    python_requires=">=3.8",
)
