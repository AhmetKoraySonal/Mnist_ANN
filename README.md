

# README
# Mnist_ANN
 Classification with Mnist Data using ANN
## Input Data Format

### Image Data
If you are providing image data for testing, the images and labels should be organized as follows:
- **Images** should be placed in a separate folder.
- **Labels** should be stored in a `.txt` file, with each label corresponding to an image in the folder.
- The label file should contain one label per line, in the same order as the images appear in the folder.

### CSV Data
If you are using a CSV file for testing, the data must follow this structure:
- The **first column** should contain the labels (class identifiers).
- The **remaining 784 columns** should contain the pixel values of the 28x28 images, flattened into a single row.
- The CSV file must either:
  - **Include a header row**, specifying column names, or
  - **Start directly from the first row**, where the first column is the label and the rest are pixel values.

Ensuring that your data follows these formats will allow smooth processing by the model.

## Command Line Arguments

### `--test_data`
**Required**  
Type: `str`  
Description: The file path to the test data. This can either be a **CSV file** containing test images and labels or a **directory** containing image files.  
Example:  
```bash
python script.py --test_data /path/to/test_data/
python script.py --test_data /path/to/test_data.csv



