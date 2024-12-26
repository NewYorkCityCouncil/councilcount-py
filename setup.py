from setuptools import setup, find_packages

setup(
    name="councilcount",
    version="1.0.3",
    author="Rachel Avram",
    author_email="ravram@council.nyc.gov",
    description="The `councilcount` package allows easy access to ACS population data across various geographic boundaries. For the boundaries that are not native to the ACS, such as council districts, an estimate is provided.",
    packages=find_packages(),
    include_package_data=True,  # Ensure data files are included
    package_data={
        "councilcount": ["data/*.csv", "data/*.geojson"],  # Include data files
    },
    python_requires=">=3.9",
    install_requires=[
        'certifi==2024.8.30',
        'geopandas==1.0.1',
        'numpy==1.26.4',
        'packaging==24.2',
        'pandas==2.2.3',
        'pyogrio==0.10.0',
        'pyproj==3.6.1',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.2',
        'shapely==2.0.6',
        'six==1.16.0',
        'tzdata==2024.2'
    ],
)
