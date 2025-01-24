import subprocess
import sys
from setuptools import setup, find_packages

# try:
#     subprocess.check_call(["gdal-config", "--version"])
# except FileNotFoundError:
#     sys.stderr.write(
#         "Error: GDAL is required to install this package. "
#         "Please install GDAL and ensure gdal-config is available in your PATH.\n"
#     )
#     sys.exit(1)

setup(
    name="councilcount",
    version="2.0.1",
    author="Rachel Avram",
    author_email="ravram@council.nyc.gov",
    description="The `councilcount` package allows easy access to ACS population data across various geographic boundaries. For the boundaries that are not native to the ACS, such as council districts, an estimate is provided. Visit https://github.com/NewYorkCityCouncil/councilcount-py/tree/main to review available functions.",
    packages=find_packages(),
    include_package_data=True,  # Ensure data files are included
    package_data={
        "councilcount": ["data/*.csv", "data/*.geojson"],  # Include data files
    },
    python_requires=">=3.9",
    install_requires=[
        'certifi==2024.12.14',
        'charset-normalizer==3.4.1',
        'geojson==3.2.0',
        'idna==3.10',
        'numpy==1.26.4', #'numpy==2.2.2',
        'pandas==2.2.3',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.2',
        'requests==2.32.3',
        'six==1.17.0',
        'tzdata==2025.1',
        'urllib3==2.3.0'
        # 'geojson==3.2.0',
        # 'numpy==1.26.4', #'numpy==2.2.1',
        # 'pandas==2.2.3',
        # 'python-dateutil==2.9.0.post0',
        # 'pytz==2024.2',
        # 'six==1.17.0',
        # 'tzdata==2024.2'
    ],
)

