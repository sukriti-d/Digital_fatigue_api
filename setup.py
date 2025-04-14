from setuptools import setup, find_packages

setup(
    name="digital_fatigue",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "scikit-learn==1.3.2",
        "pandas",
        "pydantic",
        "joblib"
    ],
) 