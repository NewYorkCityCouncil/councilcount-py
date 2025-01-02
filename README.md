# Installation

To install `councilcount` for Python, please use the following code:

``` bash
pip install councilcount
```

``` python
import councilcount as cc
```

# Prerequisites

- Python version 3.9 or above is needed.
- GDAL

## GDAL Installation Instructions

To use the `councilcount` package, GDAL must be installed on your system. Please follow the instructions below based on your operating system.

### macOS
```bash
brew install gdal
```

### Linux
```bash
sudo apt-get install gdal-bin libgdal-dev
```

### Windows
1. Visit the [GDAL binaries page](https://www.gisinternals.com/release.php).
2. Download the appropriate GDAL installer for your system.
3. Follow the installation instructions provided on the website.

---

### Notes:
- Ensure that `gdal-config` is available in your system's PATH after installation. You can verify this by running:

  ```bash
  gdal-config --version
  ```
- If you encounter issues, consult the GDAL documentation or check your system's package manager for updates.


