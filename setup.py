from setuptools import setup, find_packages

setup(
    name="councilcount",
    version="2.0.5",
    description="The `councilcount` package allows easy access to ACS population data across various NYC geographic boundaries. For the boundaries that are not native to the ACS, such as council districts, an estimate is provided.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Rachel Avram",
    author_email="datainfo@council.nyc.gov",
    license="MIT",
    url="https://github.com/NewYorkCityCouncil/councilcount-py",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "councilcount": ["data-council-logo.png", "data/*.csv", "data/*.geojson"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "certifi==2024.12.14",
        "charset-normalizer==3.4.1",
        "geojson==3.2.0",
        "idna==3.10",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.2",
        "requests==2.32.3",
        "six==1.17.0",
        "tzdata==2025.1",
        "urllib3==2.3.0",
    ],
    python_requires=">=3.9",
    # Include any other URLs such as BugTracker, Documentation if needed
    project_urls={
        "Bug Tracker": "https://github.com/NewYorkCityCouncil/councilcount-py/issues",
    },
)
