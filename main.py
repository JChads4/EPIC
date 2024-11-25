import sys
sys.path.append("..")  # Adding the parent directory to the system path
import numpy as np
from tabulate import tabulate
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import xml.etree.ElementTree as ET
from scipy.optimize import curve_fit
from scipy.stats import norm
import glob
import argparse
try:
    import scienceplots
    plt.style.use(["science"])
except ModuleNotFoundError:
    print('Tried to import SciencePlots. Not found.')

############################### BACKGROUND FUNCTIONS #################################################

def background(energy, a, b, c):
    energy_safe = np.where(energy == 0, 1, energy)  # Avoid division by zero
    return a * (1 - np.sqrt(b / energy_safe)) * np.exp(c * energy)

def draw_points(energy, counts, output_path, filename):
    fig, ax = plt.subplots()
    ax.set_title('Right-click to draw points. Click mousewheel to save.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.step(energy, counts, where='mid', label='Data', color='k', alpha=0.6)
    points = []

    def onclick(event):
        nonlocal points
        if event.button == 3:  # Right-click to add a point
            ax.plot(event.xdata, event.ydata, 'ro')
            points.append((event.xdata, event.ydata))
            fig.canvas.draw()
        elif event.button == 2:  # Mousewheel click to save the points
            save_points(points, output_path, filename)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.set_ylim(bottom=0)
    plt.show()

    return points

def save_points(points, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    filename = filename.rstrip('.dat')  # formatting
    with open(os.path.join(output_path, f'{filename}_background_points.dat'), 'w') as f:
        for point in points:
            f.write(f"{point[0]:.1f} {point[1]:.1f}\n")
    print(f"Points saved to '{os.path.join(output_path, f'{filename}_background_points.dat')}'")

def load_bkg_data(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]

def generate_background_points(file_path, output_path, filename):
    energy, counts = load_bkg_data(file_path)
    points = draw_points(energy, counts, output_path, filename)
    if points:
        save_points(points, output_path, filename)
        print(f"Background saved to {output_path}/{filename}")

############################### Code functions ######################################################
def load_data(folder, filename):
    data_file = os.path.join(folder, filename)
    return np.loadtxt(data_file)

def fwhm(energy, m, c):
    return m * energy + c

def elec_efficiency(energy, a, b, c, d, e):
    x = np.log(energy / 320)
    eff = np.exp(a + (b * x) + (c * x**2) + (d * x**3) + (e * x**4))
    return eff / 100

def gamma_efficiency(energy, a, b, c, d, e):
    x = np.log(energy / 320)
    eff = np.exp(a + (b * x) + (c * x**2) + (d * x**3) + (e * x**4))
    return eff

def err_elec_efficiency(eff):
    return 0.1 * eff

def err_gamma_efficiency(eff):
    return 0.1 * eff

def gaussian(x, area, mean, sigma):
    return area * norm.pdf(x, loc=mean, scale=sigma)

def compute_error_bands(popt, pcov, background_polynomial, e_range, optimal_curve, num_curves=100):
    """
    Compute error bands for a given fitted curve.

    Parameters:
        popt (array): Optimal parameters of the fitted curve.
        pcov (array): Covariance matrix of the fitted curve.
        background_polynomial (function): Function representing the background polynomial.
        e_range (array): Range of energy values.
        optimal_curve (array): The optimal curve.
        num_curves (int, optional): Number of curves for error band. Default is 100.

    Returns:
        std_dev (array): Standard deviation of the error bands.
    """
    curve_samples = np.random.multivariate_normal(popt, pcov, num_curves)
    error_bands = np.array([background_polynomial(e_range, *params) for params in curve_samples])
    std_dev = np.sqrt(np.sum((error_bands - optimal_curve)**2, axis=0) / (num_curves - 1))
    return std_dev

############################### BrIccs Analysis Functions #############################################

def calculate_conversion_coefficients(element, energy, multipolarity, delta):
    """
    Calculate the conversion coefficients using BrIcc for the specified energies.

    Parameters:
    element: Element symbol (e.g., 'Fm')
    energy (float): Energy of the transition.
    multipolarity (str): Multipolarity of the transition (e.g., 'E2', 'M1+E2').
    delta (float): Degree of mixing for M1+E2 transitions.

    Returns:
    dict: Dictionary with total conversion coefficients and electron energies for each energy and transition type.
    """
    shells = ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']
    alphas = {shell: {'alpha': 0.0, 'Eic': 0.0} for shell in shells}

    if multipolarity == 'E2':
        cmd = f'briccs -S {element} -g {energy} -L E2 -a'
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"BrIccs output for E2:\n{result.stdout}\n{result.stderr}")
        alphas.update(parse_briccs_output(result.stdout, multipolarity='E2'))

    elif multipolarity == 'M1':
        cmd = f'briccs -S {element} -g {energy} -L M1 -a'
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"BrIccs output for M1:\n{result.stdout}\n{result.stderr}")
        alphas.update(parse_briccs_output(result.stdout, multipolarity='M1'))

    elif multipolarity == 'E1':
        cmd = f'briccs -S {element} -g {energy} -L E1 -a'
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"BrIccs output for E1:\n{result.stdout}\n{result.stderr}")
        alphas.update(parse_briccs_output(result.stdout, multipolarity='E1'))

        # Run M2s just to see
        # cmd = f'briccs -S {element} -g {energy} -L E2 -a'
        # print(f"Running command: {cmd}")
        # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # # print(f"BrIccs output for E2:\n{result.stdout}\n{result.stderr}")
        # alphas.update(parse_briccs_output(result.stdout, multipolarity='E2'))

    elif multipolarity == 'M1+E2':
        cmd = f'briccs -S {element} -g {energy} -d {delta:.4f} -L M1+E2 -a'
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # print(f"BrIccs output for M1+E2:\n{result.stdout}\n{result.stderr}")
        alphas.update(parse_briccs_output(result.stdout, multipolarity='M1+E2'))

    else:
        print('Choose valied multipolarity please.')

    return alphas

def parse_briccs_output(output, multipolarity):
    """
    Parse the BrIcc XML output to extract the total conversion coefficients and electron energies for all relevant shells.

    Parameters:
    output (str): XML output from BrIcc.
    multipolarity (str): Multipolarity of the transition (e.g., 'E2', 'M1+E2').

    Returns:
    dict: Dictionary with conversion coefficients and electron energies for each shell.
    """
    root = ET.fromstring(output)
    shells = ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']

    alphas = {shell: {'alpha': 0.0, 'Eic': 0.0} for shell in shells}

    if multipolarity == 'E2':
        for purecc in root.findall('.//PureCC'):
            shell = purecc.get('Shell')
            if shell in shells:
                alphas[shell]['alpha'] = float(purecc.text.strip())
                alphas[shell]['Eic'] = float(purecc.get('Eic', 0.0))

    elif multipolarity == 'M1+E2':
        for mixedcc in root.findall('.//MixedCC'):
            shell = mixedcc.get('Shell')
            if shell in shells:
                alphas[shell]['alpha'] = float(mixedcc.text.strip())
                alphas[shell]['Eic'] = float(mixedcc.get('Eic', 0.0))

    elif multipolarity == 'M1':
        for purecc in root.findall('.//PureCC'):
            shell = purecc.get('Shell')
            if shell in shells:
                alphas[shell]['alpha'] = float(purecc.text.strip())
                alphas[shell]['Eic'] = float(purecc.get('Eic', 0.0))

    elif multipolarity == 'E1':
        for purecc in root.findall('.//PureCC'):
            shell = purecc.get('Shell')
            if shell in shells:
                alphas[shell]['alpha'] = float(purecc.text.strip())
                alphas[shell]['Eic'] = float(purecc.get('Eic', 0.0))

    return alphas


############################### Input File Parsing ######################################################

def parse_input_file(filename):
    variables = {}
    gamma_info_lines = []
    in_gamma_info = False
    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith('# Gamma info'):
                in_gamma_info = True
                continue
            if not in_gamma_info:
                line_no_comments = line.split('#')[0].strip()
                if '=' in line_no_comments:
                    key, value = line_no_comments.split('=', 1)
                    variables[key.strip()] = value.strip()
            else:
                # Read gamma info data
                line_no_comments = line.split('#')[0].strip()
                if line_no_comments:
                    gamma_info_lines.append(line_no_comments)
    # Now parse gamma info into a df
    if gamma_info_lines:
        # The first line is the header
        header = gamma_info_lines[0]
        columns = [col.strip() for col in header.split(',')]
        data = []
        for line in gamma_info_lines[1:]:
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(',')]
            data.append(values)
        gamma_info_df = pd.DataFrame(data, columns=columns)
    else:
        gamma_info_df = pd.DataFrame()
    return variables, gamma_info_df

############################### Main Function ##########################################################

def main():

    parser = argparse.ArgumentParser(description='Process input file.')
    parser.add_argument('input_filename', help='Input file name')
    args = parser.parse_args()

    # Read input file
    variables, gamma_info_df = parse_input_file(args.input_filename)

    print(gamma_info_df)

    # Extract variables
    loc = variables.get('loc', '')
    data_file = variables.get('data_file', '')
    gamma_spectrum_file = variables.get('gamma_spectrum_file', '')
    fit_range = variables.get('fit_range', '')
    elec_eff_params = variables.get('elec_eff_params', '')
    gam_eff_params = variables.get('gam_eff_params', '')
    elec_fwhm_params = variables.get('elec_fwhm_params', '')
    generate_background = variables.get('generate_background', 'False') == 'True'

    if fit_range:
        fit_range = tuple(map(float, fit_range.split(',')))
    else:
        fit_range = None

    # Convert efficiency & fwhm parameters to lists of floats
    elec_eff_params = list(map(float, elec_eff_params.split(',')))
    gam_eff_params = list(map(float, gam_eff_params.split(',')))
    elec_fwhm_params = list(map(float, elec_fwhm_params.split(',')))


    gamma_info_df['energy'] = gamma_info_df['energy']
    gamma_info_df['delta'] = gamma_info_df['delta']
    gamma_info_df['gamma_int'] = gamma_info_df['gamma_int']
    gamma_info_df['err_gamma_int'] = gamma_info_df['err_gamma_int']

    # Generate background points if required
    if generate_background:
        generate_background_points(f'{loc}/{data_file}', loc, data_file)

    # Call the main analysis function
    df = analyse(gamma_info_df,
                            loc=loc,
                            data_file=data_file,
                            fit_range=fit_range,
                            elec_eff_params=elec_eff_params,
                            gam_eff_params=gam_eff_params,
                            elec_fwhm_params=elec_fwhm_params,
                            gamma_spectrum_file=gamma_spectrum_file,
                            )

############################### Analysis Function ######################################################

def analyse(level_scheme_data, loc, data_file, fit_range, elec_eff_params, gam_eff_params, elec_fwhm_params, gamma_spectrum_file=None):
    ###################################### LOAD LEVEL SCHEME ########################################

    element = level_scheme_data['nucleus']
    energy = level_scheme_data['energy']
    multipolarity = level_scheme_data['multipolarity']
    delta = level_scheme_data['delta'].astype(float)
    gamma_int = level_scheme_data['gamma_int']
    err_gamma_int = level_scheme_data['err_gamma_int']

    ########################################## RUN BRICCS CALCULATIONS ##############################

    data = []
    for elem, en, mult, d, gamma, err_gamma in zip(element, energy, multipolarity, delta, gamma_int, err_gamma_int):
        
        # Get conversion coefficients for the current state
        alphas = calculate_conversion_coefficients(elem, en, mult, d)

        # Collect data for the df
        row_data = {
            'Element': elem,
            'Energy': en,
            'Multipolarity': mult,
            'Delta': d,
            'Gamma Intensity': gamma,
            'Err Gamma Intensity': err_gamma
        }

        # Add subshell specific alphas to the row data
        for shell in alphas:
            row_data[f'Alpha {shell}'] = alphas[shell]['alpha']
            row_data[f'Eic {shell}'] = alphas[shell]['Eic']

        data.append(row_data)

    df = pd.DataFrame(data)

    subshells = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']

    numeric_columns = ['Energy', 'Gamma Intensity', 'Err Gamma Intensity', 'Alpha Tot', 'Eic Tot'] + \
                  [f'Alpha {shell}' for shell in subshells] + \
                  [f'Eic {shell}' for shell in subshells] + \
                  [f'Sigma {shell}' for shell in subshells] + \
                  [f'E_Area_meas {shell}' for shell in subshells] + \
                  [f'E_Int_meas {shell}' for shell in subshells]  

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print('########################################################')
    print('##################### BRICC RESULTS ####################')
    print('########################################################')

    totals_df = df[['Energy', 'Alpha Tot', 'Eic Tot']]
    Kshell_df = df[['Energy', 'Alpha K', 'Eic K']]
    Lshell_df = df[['Energy', 'Alpha L-tot', 'Eic L-tot', 'Alpha L1', 'Eic L1', 'Alpha L2', 'Eic L2', 'Alpha L3', 'Eic L3']]
    Mshell_df = df[['Energy', 'Alpha M-tot', 'Eic M-tot', 'Alpha M1', 'Eic M1', 'Alpha M2', 'Eic M2', 'Alpha M3', 'Eic M3', 'Alpha M4', 'Eic M4', 'Alpha M5', 'Eic M5']]
    Nshell_df = df[['Energy', 'Alpha N-tot', 'Eic N-tot', 'Alpha N1', 'Eic N1', 'Alpha N2', 'Eic N2', 'Alpha N3', 'Eic N3', 'Alpha N4', 'Eic N4', 'Alpha N5', 'Eic N5', 'Alpha N6', 'Eic N6', 'Alpha N7', 'Eic N7']]
    Oshell_df = df[['Energy', 'Alpha O-tot', 'Eic O-tot', 'Alpha O1', 'Eic O1', 'Alpha O2', 'Eic O2', 'Alpha O3', 'Eic O3', 'Alpha O4', 'Eic O4', 'Alpha O5', 'Eic O5']]
    Pshell_df = df[['Energy', 'Alpha P-tot', 'Eic P-tot', 'Alpha P1', 'Eic P1', 'Alpha P2', 'Eic P2', 'Alpha P3', 'Eic P3']]
    Qshell_df = df[['Energy', 'Alpha Q', 'Eic Q']]

    print(tabulate(totals_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Kshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Lshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Mshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Nshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Oshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Pshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Qshell_df, headers='keys', tablefmt='pretty'))

    ################################ GAMMA INTENSITY CALCULATIONS ######################################################

    print('Err Gamma Intensity')
    print(df['Err Gamma Intensity'])

    print(gam_eff_params)

    df['Gamma Eff'] = gamma_efficiency(df['Energy'], *gam_eff_params)
    df['Err Gamma Eff'] = err_gamma_efficiency(df['Gamma Eff'])

    gamma_eff_ratio = df['Err Gamma Eff'] / df['Gamma Eff'].replace(0, np.nan)
    print('GAMMAS EFF RATIO')
    print(gamma_eff_ratio)
    gamma_int_ratio = df[f'Err Gamma Intensity'] / df['Gamma Intensity'].replace(0, np.nan)
    print('Gamma Int ratio')
    print(gamma_int_ratio)

    df['Gamma Int Emit'] = df['Gamma Intensity'] / df['Gamma Eff']
    df['Err Gamma Int Emit'] = df['Gamma Int Emit'] * (np.array(gamma_eff_ratio)**2 + np.array(gamma_int_ratio)**2)**0.5
    
    ################################################# ELECTRON INTENSITY CALCULATIONS ######################################################

    # Calculate electron intensities
    new_columns = {}
    for shell in ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']:
        new_columns[f'Electron Eff Alpha {shell}'] = elec_efficiency(df[f'Eic {shell}'], *elec_eff_params)
        new_columns[f'Err Electron Eff {shell}'] = err_elec_efficiency(new_columns[f'Electron Eff Alpha {shell}'])
        new_columns[f'E_Int_meas {shell}'] = df['Gamma Intensity'] * df[f'Alpha {shell}'] * new_columns[f'Electron Eff Alpha {shell}'] / df['Gamma Eff']
        elec_eff_ratio = new_columns[f'Err Electron Eff {shell}'] / new_columns[f'Electron Eff Alpha {shell}'].replace(0, np.nan)
        # Calculate the error
        new_columns[f'Err E Int meas {shell}'] = new_columns[f'E_Int_meas {shell}'] * (gamma_eff_ratio**2 + elec_eff_ratio**2 + gamma_int_ratio**2)**(0.5)

        # Calculate emitted electrons
        new_columns[f'E_Int_emit {shell}'] = new_columns[f'E_Int_meas {shell}'] / new_columns[f'Err Electron Eff {shell}']

    ######################################## FWHM CALCULATIONS ########################################################

    # Calculate widths & add to new_columns
    for shell in ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']:
        new_columns[f'Sigma {shell}'] = fwhm(df[f'Eic {shell}'],*elec_fwhm_params) /(2 * np.sqrt(2 * np.log(2)))                                                                           

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    ######################################## ELECTRON INTENSITY RESULTS ################################################

    print('########################################################')
    print('############## MEASURED INTENSITY RESULTS ##############')
    print('########################################################')

    def format_df(df, columns):
        for column in columns:
            df[column] = df[column].map('{:.2f}'.format)
        return df

    totals_df = format_df(df[['Energy', 'E_Int_meas Tot']], ['Energy', 'E_Int_meas Tot'])

    Kshell_df = format_df(df[['Energy', 'E_Int_meas K']], ['Energy', 'E_Int_meas K'])
    Lshell_df = format_df(df[['Energy', 'E_Int_meas L-tot', 'E_Int_meas L1', 'E_Int_meas L2', 'E_Int_meas L3']],
                        ['Energy', 'E_Int_meas L-tot', 'E_Int_meas L1', 'E_Int_meas L2', 'E_Int_meas L3'])
    Mshell_df = format_df(df[['Energy', 'E_Int_meas M-tot', 'E_Int_meas M1', 'E_Int_meas M2', 'E_Int_meas M3', 'E_Int_meas M4', 'E_Int_meas M5']],
                        ['Energy', 'E_Int_meas M-tot', 'E_Int_meas M1', 'E_Int_meas M2', 'E_Int_meas M3', 'E_Int_meas M4', 'E_Int_meas M5'])
    Nshell_df = format_df(df[['Energy', 'E_Int_meas N-tot', 'E_Int_meas N1', 'E_Int_meas N2', 'E_Int_meas N3', 'E_Int_meas N4', 'E_Int_meas N5', 'E_Int_meas N6','E_Int_meas N7']],
                        ['Energy', 'E_Int_meas N-tot', 'E_Int_meas N1', 'E_Int_meas N2', 'E_Int_meas N3', 'E_Int_meas N4', 'E_Int_meas N5', 'E_Int_meas N6','E_Int_meas N7'])
    Oshell_df = format_df(df[['Energy', 'E_Int_meas O-tot', 'E_Int_meas O1', 'E_Int_meas O2', 'E_Int_meas O3', 'E_Int_meas O4','E_Int_meas O5']],
                        ['Energy', 'E_Int_meas O-tot', 'E_Int_meas O1', 'E_Int_meas O2', 'E_Int_meas O3', 'E_Int_meas O4','E_Int_meas O5'])
    Pshell_df = format_df(df[['Energy', 'E_Int_meas P-tot', 'E_Int_meas P1', 'E_Int_meas P2', 'E_Int_meas P3']],
                        ['Energy', 'E_Int_meas P-tot', 'E_Int_meas P1', 'E_Int_meas P2', 'E_Int_meas P3'])
    Qshell_df = format_df(df[['Energy', 'E_Int_meas Q']], ['Energy', 'E_Int_meas Q'])

    print(tabulate(totals_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Kshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Lshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Mshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Nshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Oshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Pshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Qshell_df, headers='keys', tablefmt='pretty'))

    
    ######################################### PLOT GAMMA AND ELECTRON SPECTRA #########################################

    width=10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, (1.618/2)*width))  
    fig.patch.set_alpha(0.)
    plt.subplots_adjust(hspace=0.5)  
    fs = 22

    # Load experimental data and background file
    data = load_data(loc, data_file)
    bkg = load_data(loc, f'{data_file.rstrip('.dat')}_background_points.dat')
    energy, counts = data[:, 0], data[:, 1]
    counts = np.maximum(counts, 0) # replace negatives with 0s
    x_bkg, y_bkg = bkg[:, 0], bkg[:, 1]

    # Fitting background
    a, b, c = 180, 25, -0.025  # Sage model background initial guesses
    # TODO add a HV barrier variable for this.
    e_min, e_max = fit_range if fit_range else (min(energy), max(energy))
    fit_indices = (energy >= e_min) & (energy <= e_max)
    energy_fit = energy[fit_indices]
    counts_fit = counts[fit_indices]
    
    e_range = np.linspace(e_min, e_max, 10000)
    bin_width = energy[1] - energy[0]  # get binning of experimental spectrum
    # Calculate Areas by adjusting for bin width
    for shell in ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']:
        df[f'E_Area_meas {shell}'] = df[f'E_Int_meas {shell}'] * bin_width
        df[f'Err_E_Area_meas {shell}'] = df[f'E_Area_meas {shell}'] * (df[f'Err E Int meas {shell}'] /df[f'E_Int_meas {shell}'] )
    popt_r, pcov_r = curve_fit(background, x_bkg, y_bkg, p0=[a, b, c], bounds=([-np.inf, 24.99, -np.inf], [np.inf, 25.01, np.inf]))
    optimal_back_curve_r = background(e_range, *popt_r)
    std_dev_r = compute_error_bands(popt_r, pcov_r, background, e_range, optimal_back_curve_r)

    # Plot experimental data
    plt.step(energy, counts, where='pre', label='Data', color='k', alpha=0.6)
    plt.errorbar(energy - bin_width / 2, counts, yerr=np.sqrt(counts + 1), fmt='.', label='Data', color='k', alpha=0.2)

    # save background subtracted spectrum
    bsub_counts = counts - background(energy, *popt_r)
    file_path = loc + '/bsub_spectra.dat'
    with open(file_path, 'w') as f:
        for a, b in zip(energy, bsub_counts):
            f.write(f"{a} {b}\n")
    print(f"Background subtracted spectrum saved to {file_path}")

    num_transitions = len(df)
    cmap = plt.get_cmap('tab20', num_transitions)
    transition_colors = {index: cmap(index) for index in range(num_transitions)}

    # TODO have if statement for labels or not labels etc...

    gamma_bin_width = None
    # Plot gamma spectrum if provided
    if gamma_spectrum_file:
        gamma_data = load_data(loc, gamma_spectrum_file)
        gamma_energy, gamma_counts = gamma_data[:, 0], gamma_data[:, 1]

        gamma_bin_width = gamma_energy[1] - gamma_energy[0]
        ax1.step(gamma_energy, gamma_counts, where='pre', label='Experimental Data', color='k', alpha=0.6)
        ax1.set_ylabel(f'Counts/ {gamma_bin_width:.0f} keV', fontsize=fs-2, fontweight='bold')
        ax1.set_xlabel(f'Gamma Energy (keV)', fontsize=fs-2, fontweight='bold')
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(0, 500)
        ax1.set_title('Gammas', fontsize=fs)

        # Add labels for gamma energies
        for index, row in df.iterrows():
            gamma_energy = row['Energy']
            if row['Gamma Intensity'] == 0:
                continue
            color = transition_colors[index]  # Use the same color as in ax2

            if gamma_bin_width:
                grange = 2
                gamma_counts_region = gamma_counts[(gamma_energy - grange >= gamma_energy) & (gamma_energy + grange <= gamma_energy)]
                mask = (gamma_energy - grange <= gamma_energy) & (gamma_energy + grange >= gamma_energy)
                # Find the index of the gamma energy in the gamma_energy array
                idx = np.abs(gamma_energy - gamma_data[:, 0]).argmin()
                # Find the max count in the range
                if idx - grange >= 0 and idx + grange < len(gamma_counts):
                    max_counts_in_range = np.max(gamma_counts[idx - grange: idx + grange])
                else:
                    max_counts_in_range = gamma_counts[idx]

                # Add the label rotated at 45 degrees on top of the peak
                ax1.text(gamma_energy, max_counts_in_range * 1.05, f'{gamma_energy:.0f}', color=color, fontsize=fs-8, fontweight='bold', ha='center',
                        rotation=45, va='bottom')           
        
    # Plot experimental data
    ax2.step(energy, counts, where='pre', label='Data', color='k', alpha=0.6)
    ax2.errorbar(energy - bin_width / 2, counts, yerr=np.sqrt(counts + 1), fmt='.', label='Data', color='k', alpha=0.2)
    
    ####################################### PEAK CREATION ##########################################################

    unique_labels = set()  # To keep track of unique labels

    # Define a dictionary for shell levels
    shell_levels = {
        'K': max(counts) * 1.05,
        'L-tot': max(counts) * 1.15,
        'M-tot': max(counts) * 1.25
    }

    for index, row in df.iterrows():
        color = transition_colors[index]  # Use the color assigned to the transition
        for shell in ['K', 'L-tot', 'M-tot']:
            eic = row[f'Eic {shell}']
            if np.isnan(eic):
                continue
            if eic < fit_range[0] or eic > fit_range[1]:
                continue

            if row['Gamma Intensity'] == 0:
                continue

            label = shell.replace('-tot', '')  # Remove '-tot' from the label
            height = shell_levels.get(shell, max(counts) + 40)  # Get the corresponding level for the shell
            ymax = height / (max(counts) * 1.45)  # Adjust ymax relative to the label height

            ax2.axvline(x=eic, linewidth=2, color=color, linestyle='dotted', ymax=ymax)  # Shorten vertical lines
        
            if (row["Energy"], label) not in unique_labels:
                ax2.text(eic, height, label, color=color, fontsize=14, fontweight='bold', ha='center')
                unique_labels.add((row["Energy"], label))

    # Create and plot Gaussian peaks based on subshell parameters
    total_curve = optimal_back_curve_r.copy()  # init the total curve
    for index, row in df.iterrows():
        color = transition_colors[index]  
        for shell in ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']:
            eic = row[f'Eic {shell}']
            area = row[f'E_Area_meas {shell}']
            sig = row[f'Sigma {shell}']

            if np.isnan(eic) or np.isnan(area) or np.isnan(sig):
                continue
            peak = gaussian(e_range, area, eic, sig)
            total_curve += peak  # Add the peak to the total curve
            ax2.plot(e_range, peak + optimal_back_curve_r, linestyle='dashed', color=color)
            ax2.fill_between(e_range, peak + optimal_back_curve_r, optimal_back_curve_r, color=color, alpha=0.3)

    # Plot the total curve
    ax2.plot(e_range, total_curve, label='Total Curve', color='k', linestyle='dashdot')
    ax2.plot(e_range, optimal_back_curve_r, linestyle='-', linewidth=1, color='midnightblue', label='Background')
    ax2.fill_between(e_range, optimal_back_curve_r - std_dev_r, optimal_back_curve_r + std_dev_r, color='midnightblue', alpha=0.2)
   

    ax2.set_xlabel('Electron Energy (keV)', fontsize=fs-2, fontweight='bold')
    ax2.set_ylabel(f'Counts/ {bin_width:.0f} keV', fontsize=fs-2, fontweight='bold')
    ax2.set_xlim(0, 500)
    ax2.set_title('Electrons', fontsize=fs)
    ax2.set_ylim(bottom=0, top=max(counts)*1.4)  
    ax1.tick_params(axis='x', labelsize=fs-6, rotation=45)
    ax1.tick_params(axis='y', labelsize=fs-6)
    ax2.tick_params(axis='x', labelsize=fs-6, rotation=45)
    ax2.tick_params(axis='y', labelsize=fs-6)

    # Show the plot
    plt.subplots_adjust(bottom=0.1, top=0.9)
    plt.show()
    plt.close(fig)

    ####################################################### GOOD FITTING ####################################################################

    subshells = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']

    # Replace NaNs in electron measured areas with 0s
    for shell in subshells:
        df[f'E_Area_meas {shell}'] = df[f'E_Area_meas {shell}'].fillna(0)

    # Model function
    def ScaleTransitions_ShiftEnergy(energy, *params):
        total_function = background(energy, *popt_r).copy()
        global_energy_shift = params[0]
        index = 1
        for i, row in df.iterrows():
            scale = params[index]
            for shell in subshells:
                sigma = row[f"Sigma {shell}"]
                area = scale * row[f'E_Area_meas {shell}']
                energy_peak = row[f"Eic {shell}"] + global_energy_shift
                total_function += gaussian(energy, area, energy_peak, sigma)
            index += 1
        return total_function

    # Init params
    initial_scales = [1.0] * len(df)
    initial_params = [0.0] + initial_scales
    lower_bounds = [-0.000001] + [0] * len(df)
    upper_bounds = [0.000001] + [np.inf] * len(df)

    print('########################################################')
    print('################## FITTING CHECKS ######################')
    print('########################################################')

    # Print the initial parameters to debug
    print("Initial Parameters:")
    print("Initial Scales:", initial_scales)
    print("Initial Params:", initial_params)
    print("Lower Bounds:", lower_bounds)
    print("Upper Bounds:", upper_bounds)

    # Print the data to be fitted to debug
    print("Energy Fit:", energy_fit)
    print("Counts Fit:", counts_fit)

    # Ensure data is finite
    energy_fit = np.nan_to_num(energy_fit, nan=0.0, posinf=0.0, neginf=0.0)
    counts_fit = np.nan_to_num(counts_fit, nan=0.0, posinf=0.0, neginf=0.0)

    # Check the output of the model function with initial parameters
    initial_model_output = ScaleTransitions_ShiftEnergy(energy_fit, *initial_params)
    print("Initial Model Output:", initial_model_output)

    # Ensure the initial model output is finite
    if not np.all(np.isfinite(initial_model_output)):
        print("Initial model output contains non-finite values.")
    else:
        # Curve fit
        popt_total, pcov_total = curve_fit(ScaleTransitions_ShiftEnergy, energy_fit, counts_fit, p0=initial_params, bounds=(lower_bounds, upper_bounds))
        optimal_total_curve = ScaleTransitions_ShiftEnergy(e_range, *popt_total)

        global_energy_shift = popt_total[0]
        err_e_shift = pcov_total[0,0]**0.5
        scales = popt_total[1:]
        scales_errors = np.sqrt(np.diag(pcov_total)[1:])
        print(f'E shift = {global_energy_shift:.2f} +/- {err_e_shift:.2f}')
        print(f'Scales: {scales}')
        print(f'Scale errs: {scales_errors}')
        df['Scale'] = scales
        df['Scale Err'] = scales_errors

        # init cols for fitted areas and errors
        initial_columns = {f'Fitted_Area {shell}': np.zeros(len(df)) for shell in subshells}
        initial_columns.update({f'Err_Fitted_Area {shell}': np.zeros(len(df)) for shell in subshells})

        # Add new columns to the df
        df = pd.concat([df, pd.DataFrame(initial_columns)], axis=1)

        # Update df with fitted areas and their errors
        for i in range(len(df)):
            for shell in subshells:
                fitted_area = scales[i] * df.at[i, f'E_Area_meas {shell}']
                df.at[i, f'Fitted_Area {shell}'] = fitted_area
                # Calculate the error in the fitted area
                meas_area_err_ratio = (df.at[i, f'Err_E_Area_meas {shell}'] / df.at[i, f'E_Area_meas {shell}']) if df.at[i, f'E_Area_meas {shell}'] != 0 else 0.0
                scale_err_ratio = (scales_errors[i] / scales[i]) if scales[i] != 0 else 0.0
                err_area = fitted_area * np.sqrt(meas_area_err_ratio ** 2 + scale_err_ratio ** 2)
                # Ensure err_area is a scalar if it's a series
                if isinstance(err_area, pd.Series):
                    err_area = err_area.iloc[0]
                df.at[i, f'Err_Fitted_Area {shell}'] = err_area

    width=10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, (1.618/2)*width))  
    fig.patch.set_alpha(0.)
    plt.subplots_adjust(hspace=0.5) 
    fs = 22

    # Get unique number of transitions
    num_transitions = len(df)

    # Create a color map
    cmap = plt.get_cmap('tab20', num_transitions)

    # Define the colors for each transition
    transition_colors = {index: cmap(index) for index in range(num_transitions)}

    # Determine the binning of the gamma spectrum
    gamma_bin_width = None

    # Plot gamma spectrum if provided
    if gamma_spectrum_file:
        gamma_data = load_data(loc, gamma_spectrum_file)
        gamma_energy, gamma_counts = gamma_data[:, 0], gamma_data[:, 1]
        gamma_bin_width = gamma_energy[1] - gamma_energy[0]
        ax1.step(gamma_energy, gamma_counts, where='pre', label='Experimental Data', color='k', alpha=0.6)
        ax1.set_ylabel(f'Counts/ {gamma_bin_width:.0f} keV', fontsize=fs-2, fontweight='bold')
        ax1.set_xlabel(f'Gamma Energy (keV)', fontsize=fs-2, fontweight='bold')
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(0, 500)
        ax1.set_title('Gammas', fontsize=fs)


        for index, row in df.iterrows():
            gamma_energy = row['Energy']
            color = transition_colors[index]  # Use the same color as in ax2

            # Shade region pm 3 keV around each gamma peak
            if gamma_bin_width:
                gamma_counts_region = gamma_counts[(gamma_energy - 3 <= gamma_energy) & (gamma_energy + 3 >= gamma_energy)]

                mask = (gamma_energy - grange <= gamma_energy) & (gamma_energy + grange >= gamma_energy)
                # Find the index of the gamma energy 
                idx = np.abs(gamma_energy - gamma_data[:, 0]).argmin()
                # Find the max count in the range
                if idx - grange >= 0 and idx + grange < len(gamma_counts):
                    max_counts_in_range = np.max(gamma_counts[idx - grange: idx + grange])
                else:
                    max_counts_in_range = gamma_counts[idx]

                # Add the label rotated at 45 degrees on top of the peak
                ax1.text(gamma_energy, max_counts_in_range * 1.05, f'{gamma_energy:.0f}', color=color, fontsize=fs-8, fontweight='bold', ha='center',
                        rotation=45, va='bottom')

    # Plot experimental data with the fitted total curve
    ax2.step(energy, counts, where='pre', label='Data', color='k', alpha=0.6)
    ax2.errorbar(energy - bin_width / 2, counts, yerr=np.sqrt(counts + 1), fmt='.', label='Data', color='k', alpha=0.2)
    ax2.plot(e_range, optimal_total_curve, label='Fitted Total Curve', color='red', linestyle='solid')
    ax2.fill_between(e_range, optimal_total_curve - std_dev_r, optimal_total_curve + std_dev_r, color='r', alpha=0.1)

    ####################################### PEAK CREATION ##########################################################

    unique_labels = set()  

   
    shell_levels = {
        'K': max(counts) * 1.05,
        'L-tot': max(counts) * 1.15,
        'M-tot': max(counts) * 1.25
    }

    for index, row in df.iterrows():
        color = transition_colors[index]  
        for shell in ['K', 'L-tot', 'M-tot']:
            eic = row[f'Eic {shell}'] + global_energy_shift 
            if np.isnan(eic):
                continue
            if eic < fit_range[0] or eic > fit_range[1]:
                continue
            
            label = shell.replace('-tot', '') 
            height = shell_levels.get(shell, max(counts) + 40)  
            ymax = height / (max(counts) * 1.45)  

            # ax2.axvline(x=eic, linewidth=2, color=color, linestyle='dotted', ymax=ymax)  # Shorten vertical lines

            if (row["Energy"], label) not in unique_labels:
                # ax2.text(eic, height, label, color=color, fontsize=14, fontweight='bold', ha='center')
                unique_labels.add((row["Energy"], label))

    # Plot new rescaled intensities
    for index, row in df.iterrows():
        color = transition_colors[index]
        for shell in ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']:
            eic = row[f'Eic {shell}'] + global_energy_shift  # Apply the energy shift here - set to 0 at the moment.
            area = scales[index] * row[f'E_Area_meas {shell}']  # Apply the scaling factor
            sig = row[f'Sigma {shell}']

            if np.isnan(eic) or np.isnan(area) or np.isnan(sig):
                continue
            peak = gaussian(e_range, area, eic, sig)
            plot_range_mask = (e_range >= eic - 1) & (e_range <= eic + 1)
            ax2.plot(e_range, peak + optimal_back_curve_r, linestyle='dashed', color=color)
            ax2.fill_between(e_range, peak + optimal_back_curve_r, optimal_back_curve_r, color=color, alpha=0.3)

    # Set axis labels and grid
    ax2.plot(e_range, optimal_back_curve_r, linestyle='-', linewidth=1, color='midnightblue')
    ax2.set_xlabel('Electron Energy (keV)', fontsize=fs-2, fontweight='bold')
    ax2.set_ylabel(f'Counts/ {bin_width:.0f} keV', fontsize=fs-2, fontweight='bold')
    ax2.set_xlim(0, 500)
    ax2.set_title('Electrons', fontsize=fs)
    ax2.set_ylim(bottom=0, top=max(counts) * 1.4)  # Adjust top limit to make room for labels

    ax1.tick_params(axis='x', labelsize=fs-8, rotation=45)
    ax2.tick_params(axis='x', labelsize=fs-8, rotation=45)

    # Goodness of fit calculation + display in legend
    # Calculate chi-squared
    # Avoid division by zero by adding a small value (1) where counts_fit is 0
    counts_fit = np.where(counts_fit == 0, 1, counts_fit)
    chi_squared = np.sum(((counts_fit - ScaleTransitions_ShiftEnergy(energy_fit, *popt_total)) ** 2) / counts_fit)
    dof = len(counts_fit) - len(popt_total)
    rchisq = chi_squared / dof
    print(f"Chi-squared: {chi_squared:.2f}, Degrees of Freedom: {dof}, Reduced Chi-squared: {rchisq:.2f}")
    ax2.plot([], [], ' ', label=f'chisq/DOF = {rchisq:.2f}')
    ax2.legend(fontsize='large', loc='upper right')
    ax1.tick_params(axis='x', labelsize=fs-6, rotation=45)
    ax1.tick_params(axis='y', labelsize=fs-6)
    ax2.tick_params(axis='x', labelsize=fs-6, rotation=45)
    ax2.tick_params(axis='y', labelsize=fs-6)
    plt.show()
    plt.close(fig)


    print('########################################################')
    print('################## FITTING RESULTS #####################')
    print('########################################################')

    for shell in subshells:
        df[f'Fitted_Int {shell}'] =  df[f'Fitted_Area {shell}']/ bin_width
        df[f'Err Fitted Int {shell}'] = df[f'Fitted_Int {shell}'] * df[f'Err_Fitted_Area {shell}'] / df[f'Fitted_Area {shell}']

        # calculate the emitted fitted intensity
        df[f'Fitted Int Emit {shell}'] = df[f'Fitted_Int {shell}'] / elec_efficiency(df[f'Eic {shell}'], *elec_eff_params)
        df[f'Err Fitted Int Emit {shell}'] = df[f'Fitted Int Emit {shell}']  * (df[f'Err Fitted Int {shell}'] / df[f'Fitted_Int {shell}'] )

    def format_df(df, columns):
        for column in columns:
            df[column] = df[column].map('{:.2f}'.format)
        return df

    Kshell_df = format_df(df[['Energy', 'Fitted_Int K']], ['Energy', 'Fitted_Int K'])
    Lshell_df = format_df(df[['Energy', 'Fitted_Int L1', 'Fitted_Int L2', 'Fitted_Int L3']],
                        ['Energy','Fitted_Int L1', 'Fitted_Int L2', 'Fitted_Int L3'])
    Mshell_df = format_df(df[['Energy', 'Fitted_Int M1', 'Fitted_Int M2', 'Fitted_Int M3', 'Fitted_Int M4', 'Fitted_Int M5']],
                        ['Energy', 'Fitted_Int M1', 'Fitted_Int M2', 'Fitted_Int M3', 'Fitted_Int M4', 'Fitted_Int M5'])
    Nshell_df = format_df(df[['Energy','Fitted_Int N1', 'Fitted_Int N2', 'Fitted_Int N3', 'Fitted_Int N4', 'Fitted_Int N5', 'Fitted_Int N6','Fitted_Int N7']],
                        ['Energy',  'Fitted_Int N1', 'Fitted_Int N2', 'Fitted_Int N3', 'Fitted_Int N4', 'Fitted_Int N5', 'Fitted_Int N6','Fitted_Int N7'])
    Oshell_df = format_df(df[['Energy',  'Fitted_Int O1', 'Fitted_Int O2', 'Fitted_Int O3', 'Fitted_Int O4','Fitted_Int O5']],
                        ['Energy',  'Fitted_Int O1', 'Fitted_Int O2', 'Fitted_Int O3', 'Fitted_Int O4','Fitted_Int O5'])
    Pshell_df = format_df(df[['Energy', 'Fitted_Int P1', 'Fitted_Int P2', 'Fitted_Int P3']],
                        ['Energy', 'Fitted_Int P1', 'Fitted_Int P2', 'Fitted_Int P3'])
    Qshell_df = format_df(df[['Energy', 'Fitted_Int Q']], ['Energy', 'Fitted_Int Q'])


    print(tabulate(Kshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Lshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Mshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Nshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Oshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Pshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Qshell_df, headers='keys', tablefmt='pretty'))

    ############################################## CONVERSION COEFFICIENT CALCULATION #########################################################
    
    # SUBSHELLS

    new_columns = {}
    for shell in subshells:
        
        new_columns[f'Meas_alpha_{shell}'] = df[f'Fitted Int Emit {shell}']/ df['Gamma Int Emit'] 
        new_columns[f'Err Meas_alpha_{shell}'] =  new_columns[f'Meas_alpha_{shell}'] * ((df[f'Err Fitted Int Emit {shell}']/df[f'Fitted Int Emit {shell}'])**2 + (df['Err Gamma Int Emit'] /df['Gamma Int Emit'] )**2)**(0.5)
    
        # Debug: Print to check if the new columns are formed correctly
        print(f"### DEBUG SUBSHELLS {shell} ###")
        print(f"Meas_alpha_{shell}:")
        print(new_columns[f'Meas_alpha_{shell}'].head())
        print(f"Err Meas_alpha_{shell}:")
        print(new_columns[f'Err Meas_alpha_{shell}'].head())



    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    for shell in subshells:
        if df[f'Meas_alpha_{shell}'].shape == df[f'Err Meas_alpha_{shell}'].shape:
            print(f"Subshell {shell}: Dimensions match. YEEEEEEEEEEEESSSSSSSSSSSSSSSSSSSSSS")
        else:
            print(f"Subshell {shell}: Dimension mismatch. NOOOOOOOOOOOOOOOOOOO")

    # print('################## DEBUG SUBSHELL CALC #######################')
    # # Print values for the first row only
    # for shell in subshells:
    #     print(f'######## STARTING SHELL {shell} ##############')
    #     print("Fitted_Int", df.loc[0, f'Fitted_Int {shell}'])
    #     print("Err Fitted Int", df.loc[0, f'Err Fitted Int {shell}'])
    #     print("Gamma Intensity", df.loc[0, 'Gamma Intensity'])
    #     print("Err Gamma Intensity", df.loc[0, 'Err Gamma Intensity'])
    #     print("Meas_alpha", df.loc[0, f'Meas_alpha_{shell}'])
    #     print("Err Meas_alpha", df.loc[0, f'Err Meas_alpha_{shell}'])
    #     print('######### NEXT SHELL ##############')

    print('##########################################################################')
    print('################## CONVERSION COEFF SUBSHELL RESULTS #####################')
    print('##########################################################################')

    # Subshell results
    Kshell_df = format_df(df[['Energy', 'Meas_alpha_K']], ['Energy', 'Meas_alpha_K'])
    Lshell_df = format_df(df[['Energy', 'Meas_alpha_L1', 'Meas_alpha_L2', 'Meas_alpha_L3']],
                        ['Energy','Meas_alpha_L1', 'Meas_alpha_L2', 'Meas_alpha_L3'])
    Mshell_df = format_df(df[['Energy', 'Meas_alpha_M1', 'Meas_alpha_M2', 'Meas_alpha_M3', 'Meas_alpha_M4', 'Meas_alpha_M5']],
                        ['Energy', 'Meas_alpha_M1', 'Meas_alpha_M2', 'Meas_alpha_M3', 'Meas_alpha_M4', 'Meas_alpha_M5'])
    Nshell_df = format_df(df[['Energy','Meas_alpha_N1', 'Meas_alpha_N2', 'Meas_alpha_N3', 'Meas_alpha_N4', 'Meas_alpha_N5', 'Meas_alpha_N6','Meas_alpha_N7']],
                        ['Energy',  'Meas_alpha_N1', 'Meas_alpha_N2', 'Meas_alpha_N3', 'Meas_alpha_N4', 'Meas_alpha_N5', 'Meas_alpha_N6','Meas_alpha_N7'])
    Oshell_df = format_df(df[['Energy',  'Meas_alpha_O1', 'Meas_alpha_O2', 'Meas_alpha_O3', 'Meas_alpha_O4','Meas_alpha_O5']],
                        ['Energy',  'Meas_alpha_O1', 'Meas_alpha_O2', 'Meas_alpha_O3', 'Meas_alpha_O4','Meas_alpha_O5'])
    Pshell_df = format_df(df[['Energy', 'Meas_alpha_P1', 'Meas_alpha_P2', 'Meas_alpha_P3']],
                        ['Energy', 'Meas_alpha_P1', 'Meas_alpha_P2', 'Meas_alpha_P3'])
    Qshell_df = format_df(df[['Energy', 'Meas_alpha_Q']], ['Energy', 'Meas_alpha_Q'])

    # Subshell results
    # Kshell_df = format_df(df[['Energy', 'Meas_alpha_K', 'Err Meas_alpha_K']], ['Energy', 'Meas_alpha_K', 'Err Meas_alpha_K'])
    # Lshell_df = format_df(df[['Energy', 'Meas_alpha_L1', 'Meas_alpha_L2', 'Meas_alpha_L3', 'Err Meas_alpha_L1', 'Err Meas_alpha_L2', 'Err Meas_alpha_L3']],
    #                     ['Energy', 'Meas_alpha_L1', 'Meas_alpha_L2', 'Meas_alpha_L3', 'Err Meas_alpha_L1', 'Err Meas_alpha_L2', 'Err Meas_alpha_L3'])
    # Mshell_df = format_df(df[['Energy', 'Meas_alpha_M1', 'Meas_alpha_M2', 'Meas_alpha_M3', 'Meas_alpha_M4', 'Meas_alpha_M5', 'Err Meas_alpha_M1', 'Err Meas_alpha_M2', 'Err Meas_alpha_M3', 'Err Meas_alpha_M4', 'Err Meas_alpha_M5']],
    #                     ['Energy', 'Meas_alpha_M1', 'Meas_alpha_M2', 'Meas_alpha_M3', 'Meas_alpha_M4', 'Meas_alpha_M5', 'Err Meas_alpha_M1', 'Err Meas_alpha_M2', 'Err Meas_alpha_M3', 'Err Meas_alpha_M4', 'Err Meas_alpha_M5'])
    # Nshell_df = format_df(df[['Energy', 'Meas_alpha_N1', 'Meas_alpha_N2', 'Meas_alpha_N3', 'Meas_alpha_N4', 'Meas_alpha_N5', 'Meas_alpha_N6', 'Meas_alpha_N7', 'Err Meas_alpha_N1', 'Err Meas_alpha_N2', 'Err Meas_alpha_N3', 'Err Meas_alpha_N4', 'Err Meas_alpha_N5', 'Err Meas_alpha_N6', 'Err Meas_alpha_N7']],
    #                     ['Energy', 'Meas_alpha_N1', 'Meas_alpha_N2', 'Meas_alpha_N3', 'Meas_alpha_N4', 'Meas_alpha_N5', 'Meas_alpha_N6', 'Meas_alpha_N7', 'Err Meas_alpha_N1', 'Err Meas_alpha_N2', 'Err Meas_alpha_N3', 'Err Meas_alpha_N4', 'Err Meas_alpha_N5', 'Err Meas_alpha_N6', 'Err Meas_alpha_N7'])
    # Oshell_df = format_df(df[['Energy', 'Meas_alpha_O1', 'Meas_alpha_O2', 'Meas_alpha_O3', 'Meas_alpha_O4', 'Meas_alpha_O5', 'Err Meas_alpha_O1', 'Err Meas_alpha_O2', 'Err Meas_alpha_O3', 'Err Meas_alpha_O4', 'Err Meas_alpha_O5']],
    #                     ['Energy', 'Meas_alpha_O1', 'Meas_alpha_O2', 'Meas_alpha_O3', 'Meas_alpha_O4', 'Meas_alpha_O5', 'Err Meas_alpha_O1', 'Err Meas_alpha_O2', 'Err Meas_alpha_O3', 'Err Meas_alpha_O4', 'Err Meas_alpha_O5'])
    # Pshell_df = format_df(df[['Energy', 'Meas_alpha_P1', 'Meas_alpha_P2', 'Meas_alpha_P3', 'Err Meas_alpha_P1', 'Err Meas_alpha_P2', 'Err Meas_alpha_P3']],
    #                     ['Energy', 'Meas_alpha_P1', 'Meas_alpha_P2', 'Meas_alpha_P3', 'Err Meas_alpha_P1', 'Err Meas_alpha_P2', 'Err Meas_alpha_P3'])
    # Qshell_df = format_df(df[['Energy', 'Meas_alpha_Q', 'Err Meas_alpha_Q']], ['Energy', 'Meas_alpha_Q', 'Err Meas_alpha_Q'])


    print(tabulate(Kshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Lshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Mshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Nshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Oshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Pshell_df, headers='keys', tablefmt='pretty'))
    print(tabulate(Qshell_df, headers='keys', tablefmt='pretty'))

    # SHELLS

    # Add in calculation for shell total alphas for each transition
    shell_groups = {
    'K-tot': ['K'],
    'L-tot': ['L1', 'L2', 'L3'],
    'M-tot': ['M1', 'M2', 'M3', 'M4', 'M5'],
    'N-tot': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'],
    'O-tot': ['O1', 'O2', 'O3', 'O4', 'O5'],
    'P-tot': ['P1', 'P2', 'P3', 'P4', 'P5'],
    'Q-tot': ['Q']
    }

    # Calculate the total alphas for each shell group
    for group_name, sub_shells in shell_groups.items():
        df[f'Meas_alpha_{group_name}'] = df[[f'Meas_alpha_{sub_shell}' for sub_shell in sub_shells]].sum(axis=1)
        df[f'Err Meas_alpha_{group_name}'] = ((df[[f'Err Meas_alpha_{sub_shell}' for sub_shell in sub_shells]] ** 2).sum(axis=1))**(0.5)

    for group_name in shell_groups.keys():
        if df[f'Meas_alpha_{group_name}'].shape == df[f'Err Meas_alpha_{group_name}'].shape:
            print(f"Shell group {group_name}: Dimensions match. ")
        else:
            print(f"Shell group {group_name}: Dimension mismatch.")


    print('#############################################################################')
    print('################## CONVERSION COEFF SHELL TOTAL RESULTS #####################')
    print('#############################################################################')

    # Renaming the BrIcc total values to the DataFrame
    for group_name in ['K', 'L-tot', 'M-tot', 'N-tot', 'O-tot', 'P-tot', 'Q']:
        df[f'Alpha_{group_name}_briccs'] = df[f'Alpha {group_name}']

    # Select and round the columns to 2 decimal places
    totals_df = df[['Energy', 'Meas_alpha_K', 'Alpha_K_briccs', 'Meas_alpha_L-tot', 'Alpha_L-tot_briccs',
                    'Meas_alpha_M-tot', 'Alpha_M-tot_briccs', 'Meas_alpha_N-tot', 'Alpha_N-tot_briccs',
                    'Meas_alpha_O-tot', 'Alpha_O-tot_briccs', 'Meas_alpha_P-tot', 'Alpha_P-tot_briccs',
                    'Meas_alpha_Q', 'Alpha_Q_briccs']].round(2)

    # Print out the results
    print(tabulate(totals_df, headers='keys', tablefmt='pretty'))
    
    ############################################## CONVERSION COEFFICIENT PLOTTING #########################################################
    def plot_conversion_coefficients(df, subshells):
        plt.figure(figsize=(10, 6))

        # Define markers for each shell
        markers = {
            'K': 's',
            'L1': 'o', 'L2': '^', 'L3': 'v',
            'M1': 'D', 'M2': '<', 'M3': '>', 'M4': 'p', 'M5': '*',
            'N1': 'P', 'N2': 'X', 'N3': 'H', 'N4': '8', 'N5': 'h', 'N6': '+', 'N7': 'x',
            'O1': '1', 'O2': '2', 'O3': '3', 'O4': '4', 'O5': 'd',
            'P1': '|', 'P2': '_', 'P3': ',', 'P4': '.', 'P5': '>',
            'Q': '1'
        }

        # Get unique number of transitions
        num_transitions = len(df)
        cmap = plt.get_cmap('tab20', num_transitions)
        transition_colors = {index: cmap(index) for index in range(num_transitions)}

        plt.figure(figsize=(10, 6))
        for index, row in df.iterrows():
            if row['Gamma Intensity'] <1:
                continue
            color = transition_colors[index]

            # Calculate l_coeff: (Meas_alpha_L1 + Meas_alpha_L2) / Meas_alpha_L3
            l1 = row[f'Meas_alpha_L1']
            l2 = row[f'Meas_alpha_L2']
            l3 = row[f'Meas_alpha_L3']

            if np.isnan(l1) or np.isnan(l2) or np.isnan(l3) or l3 == 0:
                continue
            
            l_coeff = (l1 + l2) / l3
            
            plt.scatter(row['Energy'], l_coeff, label=f'{row["Energy"]} keV', color=color, marker=markers['L1'])

            # Calculate BRICCS l_coeff: (Alpha_L1_briccs + Alpha_L2_briccs) / Alpha_L3_briccs
            l1_briccs = row[f'Alpha L1']
            l2_briccs = row[f'Alpha L2']
            l3_briccs = row[f'Alpha L3']

            if not (np.isnan(l1_briccs) or np.isnan(l2_briccs) or np.isnan(l3_briccs) or l3_briccs == 0):
                l_coeff_briccs = (l1_briccs + l2_briccs) / l3_briccs
                plt.scatter(row['Energy'], l_coeff_briccs, label=f'BRICCS L_coeff {row["Energy"]} keV', color=color, marker=markers['L3'])


        plt.yscale('log')
        plt.xlabel('Transition Energy (keV)')
        plt.ylabel('L Conversion Coefficient')
        plt.title('L Conversion Coefficient vs Transition Energy')
        plt.tight_layout()

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')

        plt.show()
        plt.close()

    ############################################# BRICCS COMPARISON PLOTTING #############################################
    # Extract data for plotting
    transition_energies = df['Energy']
    electron_energies = {
        # 'K': df['Eic K'],
        'L-tot': df['Eic L-tot'],
        'M-tot': df['Eic M-tot'],
        # 'N-tot': df['Eic N-tot'],
        # 'O-tot': df['Eic O-tot'],
        # 'P-tot': df['Eic P-tot'],
        # 'Q': df['Eic Q']
    }
    meas_alpha_totals = {
        # 'K': df['Meas_alpha_K'],
        'L-tot': df['Meas_alpha_L-tot'],
        'M-tot': df['Meas_alpha_M-tot'],
        # 'N-tot': df['Meas_alpha_N-tot'],
        # 'O-tot': df['Meas_alpha_O-tot'],
        # 'P-tot': df['Meas_alpha_P-tot'],
        # 'Q': df['Meas_alpha_Q']
    }

    err_meas_alpha_totals = {
        # 'K': df['Err Meas_alpha_K'],
        'L-tot': df['Err Meas_alpha_L-tot'],
        'M-tot': df['Err Meas_alpha_M-tot'],
        # 'N-tot': df['Err Meas_alpha_N-tot'],
        # 'O-tot': df['Err Meas_alpha_O-tot'],
        # 'P-tot': df['Err Meas_alpha_P-tot'],
        # 'Q': df['Err Meas_alpha_Q']
    }
    briccs_alpha_totals = {
        # 'K': df['Alpha_K_briccs'],
        'L-tot': df['Alpha_L-tot_briccs'],
        'M-tot': df['Alpha_M-tot_briccs'],
        # 'N-tot': df['Alpha_N-tot_briccs'],
        # 'O-tot': df['Alpha_O-tot_briccs'],
        # 'P-tot': df['Alpha_P-tot_briccs'],
        # 'Q': df['Alpha_Q_briccs']
    }

    # Define colors for each shell
    colors = {
        # 'K': 'blue',
        'L-tot': 'orange',
        'M-tot': 'green',
        # 'N-tot': 'red',
        # 'O-tot': 'purple',
        # 'P-tot': 'brown',
        # 'Q': 'pink'
    }

    # Ensure no NaNs in yerr and convert to NumPy array
    for shell in err_meas_alpha_totals:
        err_meas_alpha_totals[shell] = df[f'Err Meas_alpha_{shell}'].replace(np.nan, 0).to_numpy()

    print("### DEBUG INFO ###")
    for shell, electron_energy in electron_energies.items():
        print(f"Shell: {shell}")
        print("Electron Energy:", electron_energy.to_numpy())
        print("Meas Alpha Totals:", meas_alpha_totals[shell].to_numpy())
        print("Err Meas Alpha Totals:", err_meas_alpha_totals[shell])
        print("---------------")

    ################################################# LM Coefficient plot
    def plot_conversion_coefficients_with_arrows(df):

        width=8
        fig, ax = plt.subplots(figsize=(width, width/1.618)) 
        fig.patch.set_alpha(0.)
        plt.subplots_adjust(hspace=0.5)  
        fs = 22

        num_transitions = len(df)
        cmap = plt.get_cmap('tab20', num_transitions)
        transition_colors = {index: cmap(index) for index in range(num_transitions)}
        label_added = False
        for index, row in df.iterrows():
            if row['Gamma Intensity'] < 0:
                continue
            
            eic_L = row['Eic L-tot']
            eic_M = row['Eic M-tot']
            
            alpha_L_meas = row['Meas_alpha_L-tot']
            alpha_M_meas = row['Meas_alpha_M-tot']
            
            alpha_L_briccs = row['Alpha_L-tot_briccs']
            alpha_M_briccs = row['Alpha_M-tot_briccs']
            
            transition_color = transition_colors[index]
            
            ax.errorbar(eic_L, alpha_L_meas, yerr=row['Err Meas_alpha_L-tot'], fmt='x', color='k', label='Measured' if index == 0 else "")
            ax.errorbar(eic_M, alpha_M_meas, yerr=row['Err Meas_alpha_M-tot'], fmt='x', color='k')
            
            # Plot BRICCS data points
                # E2's
            ax.scatter(eic_L, alpha_L_briccs, marker='o', color='r', label='BRICCS E2' if index == 0 else "") # transition colour
            ax.scatter(eic_M, alpha_M_briccs, marker='o', color='r')
            ax.plot([eic_L, eic_M], [alpha_L_briccs, alpha_M_briccs], color='r', linestyle='-', linewidth=1)

               
                # M1's
            # level_scheme_data = pd.read_csv(level_scheme, delimiter=',', comment='#')
            # element, energy, multipolarity, delta, gamma_int, err_gamma_int = level_scheme_data['nucleus'], level_scheme_data['energy'], level_scheme_data['multipolarity'], level_scheme_data['delta'], level_scheme_data['gamma_int'], level_scheme_data['err_gamma_int']
            
            element = level_scheme_data['nucleus']
            energy = level_scheme_data['energy']
            
            for elem, en, colour in zip(element, energy, range(num_transitions)):

                alphas_m1 = calculate_conversion_coefficients(elem, en, 'M1', 0)
                print(' ALPHAS M! ')
                print(alphas_m1)
                ax.scatter(alphas_m1['L-tot']['Eic'], alphas_m1['L-tot']['alpha'], marker='^', color='b', label='BRICCS M1' if not label_added else "")
                ax.scatter(alphas_m1['M-tot']['Eic'], alphas_m1['M-tot']['alpha'], marker='^', color='b') # cmap(colour)

                # Plot connecting lines for m1
                ax.plot([alphas_m1['L-tot']['Eic'], alphas_m1['M-tot']['Eic']], [alphas_m1['L-tot']['alpha'], alphas_m1['M-tot']['alpha']], color='b', linestyle='-', linewidth=1)

                # plot connecting lines for 
                
                alphas_e1 = calculate_conversion_coefficients(elem, en, 'E1', 0)
                # ax.scatter(alphas_e1['L-tot']['Eic'], alphas_e1['L-tot']['alpha'], marker='s', color='g', label='BRICCS E1' if not label_added else "")
                # ax.scatter(alphas_e1['M-tot']['Eic'], alphas_e1['M-tot']['alpha'], marker='s', color='g') # cmap(colour)
                label_added = True

            pos_scale = 12
            L_ypos = max(alpha_L_briccs, alpha_L_meas) * pos_scale
            M_ypos = max(alpha_M_briccs, alpha_M_meas) * pos_scale

            L_ymin = max(alpha_L_briccs, alpha_L_meas) * pos_scale/1.5
            M_ymin =  max(alpha_M_briccs, alpha_M_meas) * pos_scale/1.5
            
            # Plot horizontal line connecting the L and M components
            ax.plot([eic_L, eic_M], [L_ypos, M_ypos], color='black', lw=1.5)

            # Plot vertical arrows pointing downwards
            ax.plot([eic_L, eic_L], [L_ypos, L_ymin], color='black', lw=1.5, marker='')
            ax.plot([eic_M, eic_M], [M_ypos, M_ymin], color='black', lw=1.5, marker='')

           # Calculate the angle for the text rotation
            angle = np.degrees(np.arctan2(np.log10(M_ypos) - np.log10(L_ypos), eic_M - eic_L))

            # Plot labels for L, M, and transition energy
            ax.text(eic_L, L_ymin * 0.45, 'L', ha='center', va='bottom', fontsize=14, fontweight='bold')
            ax.text(eic_M, M_ymin * 0.45, 'M', ha='center', va='bottom', fontsize=14, fontweight='bold')
            ax.text((eic_L + eic_M) / 2, (L_ypos*1.1), f'{row['Energy']:.0f}', ha='center', va='bottom',
                    fontsize=16, fontweight='bold', rotation=angle, color='k')



        ax.set_yscale('log')
        ax.set_xlabel('Electron energy / keV', fontsize=fs)
        ax.set_ylabel(r'$\alpha$', fontsize=fs)
        ax.set_ylim(bottom=1e-3)
        ax.legend(fontsize=fs-6, fancybox=True)
        ax.tick_params(axis='both', labelsize=fs-8)
        plt.tight_layout()
        plt.show()

    # TODO ADD the L subshell ratios

    # Plot the conversion coefficients with arrows
    plot_conversion_coefficients_with_arrows(df)
    # plot_L_subshell_ratios(df)

    ########################################## SAVE DF #################################################################
    df.to_csv(f'{loc}/FinalDF.csv', sep='\t', encoding='utf-8', index=False, header=True)

    ############################################ INTENSITY TABLE ########################################################
   
    subshells = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']
    df['Electron Int Emit'] = df[[f'Fitted Int Emit {shell}' for shell in subshells]].sum(axis=1)

    intensity_df = df[['Energy', 'Gamma Int Emit', 'Electron Int Emit', 'Alpha Tot', 'Gamma Eff', 'Err Gamma Eff']]
    def format_intensity_df(df):
        for column in df.columns:
            df[column] = df[column].map('{:.2f}'.format)
        return df

    formatted_intensity_df = format_intensity_df(intensity_df)
    print(tabulate(formatted_intensity_df, headers='keys', tablefmt='pretty'))

    intensity_df.to_csv(f'{loc}/output.csv', index=False)
    
    return df

if __name__ == "__main__":
    main()
