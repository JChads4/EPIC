
# Electron and Photon Intensity Calculator

This project is a program designed for the analysis of **conversion electron and gamma spectroscopy** data. 
## Features

-**Input File Processing**: The input file containing paths to experimental data files, optional flags, experiment parameters, and gamma-ray measurements is parsed.
- **Data Processing**: Reads and processes gamma and electron data files from experimental measurements.
- **Background Subtraction**: Allows for background correction using user-provided background data of the conversion electron spectrum.
- **Expectation** The expected conversion electron intensity for all subshells is visualised over the experimental data. These are calculated by executing BrIcc's commands, and parsing the output. User-defined parameters such as; the measured gamma-ray intensities, detector efficiencies, and electron detector energy resolution, are also used.
- **Peak Fitting**: Fits conversion electron peaks over a user-defined range, by applying a 'scaling factor' (effectively fitting a conversion coefficient) for each transition specified in the input file. Therefore the number of fitted parameters is equal the number of input transitions. The relative intensity between all electron subshells in a transition is given by BrIcc and kept constant in the fitting process.
- **Conversion Coefficient Calculation**: Calculates, and plots the measured conversion coefficients based on the scaling factors determined from the fitting procedure.
- **Results Export**: Outputs the calculated results and fitted parameters to CSV files for further analysis.

## Requirements

- **Python 3.6** or higher
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `tabulate`
  - `scienceplots` (optional, for enhanced plotting styles)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JChads4/EPIC.git
   cd EPIC
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install BrIcc**:

   The program relies on the BrIcc software to calculate conversion coefficients. Please ensure that BrIcc is installed and accessible from the command line.

   - Download BrIcc from the official website: [BrIcc Download](https://www.nndc.bnl.gov/bricc/)
   - Follow the installation instructions provided.

## Usage

1. **Prepare Input Files**:

   - **Data Files**: The electron, and gamma ascii data files in a two-column format (energy, counts).

2. **Create an Input Configuration File**:

   Create a text file (e.g., `input.txt`) with the necessary parameters:

   ```
   loc = /path/to/data
   data_file = electrons.dat
   gamma_spectrum_file = gammas.dat
   fit_range = 0, 1000
   elec_eff_params = a, b, c, d, e
   gam_eff_params = a, b, c, d, e
   elec_fwhm_params = m, c 
   generate_background = False

   # Gamma info
   nucleus, energy, multipolarity, delta, gamma_int, err_gamma_int
   U-238, 143.8, E2, 0.0, 1000, 50
   ```
   See code for the efficiency functions. Delta represent the mixing parameter.

3. **Run the Program**:

   ```bash
   python main.py input.txt
   ```

   If the generate background flag is set 'True' (which it should be for the first time running the code). The user need only right click points where the background is (by default the background is set to go to 0 at 25), and once finished close the window. The program will generate the fiteed curve data points as a txt file in the /data directory, and can be used in future by then setting the generate background flag to 'False'.

4. **View the Results**:

   - The program will generate plots showing the spectra and fitted peaks.
   - Results will be saved to CSV files in the specified location.

## Functionality Overview

### Main Functionality

The main function orchestrates the workflow:

- Parses the input configuration file.
- Loads data and allows user defined background if the flag is set True.
- Processes the level scheme data in the input file.
- Calculates conversion coefficients using BrIcc.
- Fits the spectra using Gaussian peaks.
- Optimises the fit parameters.
- Calculates fitted conversion coefficients.
- Generates plots for visualisation.
- Saves the results to output files.

### Key Functions

- `parse_input_file(filename)`: Parses the input configuration file and extracts parameters and gamma info.
- `calculate_conversion_coefficients(element, energy, multipolarity, delta)`: Uses BrIcc to calculate conversion coefficients.
- `analyse(...)`: Core function that performs the spectral fitting and analysis.
- `background(energy, a, b, c)`: Models the background for background subtraction.
- `gaussian(x, area, mean, sigma)`: Gaussian function used for peak fitting.

## Troubleshooting

- **BrIcc Not Found**: Ensure that BrIcc is installed and the executable is in your system's PATH.
- **ModuleNotFoundError**: Install any missing Python packages using `pip install package-name`.
- **Fitting Issues**: Check the initial parameters and ensure that the data files are correctly formatted.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **BrIcc**: Thanks to the developers of BrIcc for providing the tool for internal conversion coefficient calculations.
- **SciencePlots**: The `scienceplots` package is used for enhanced plotting aesthetics.

## Contact

For questions or feedback, please contact Jamie Chadderton at [jamiechadderton8@gmail.com].

---
