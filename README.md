# FourDimensionalData

Scripts to open various 4-dimensional data formats, with 2 spatial and 2 image dimensions, in a common format.
Supported data formats are currently based around Transmission Electron Microscopy and
include ASTAR .blo and TVIPS .tvips files.

As the file sizes of these types of data may be several GB, the file data is is not loaded by default.
The data is read and loaded to memory when requesting specific frames using generator parsing.

This package is based on code parsers in [Hyperspy](https://github.com/din14970/TVIPSconverter) and [TVIPSconverter](https://github.com/din14970/TVIPSconverter).

### Installation

Clone this repository, navigate to the base directory and run `pip install .`

### Usage

To open a file:

```python
import FourDimensionalData

data = FourDimensionalData.from_file("my_data.blo")
```

### Issues

Please check current and previous Issues and Pull Requests for similar issues. For new issues, please open an Issue describing the problem.

### Funding

This code was developed under the scope of ANR project [ANR-19-CE42–0017](https://anr.fr/Project-ANR-19-CE42-0017).
