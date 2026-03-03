from .model import main_model
from .utitlity_fitting import compute_covariance_and_correlation,plot_profile_likelihoods,profile_likelihood_plotter,least_square_fitting_algorithim

from .data_reading import read_excel
import pathlib

# path for experimental data
exp_path=pathlib.PureWindowsPath(r"G:\My Drive\PhD studies\PhD Project\TcellProject\publicationmaterial\experimental_data").as_posix()  #int((nhits+1)/2)

# path for saving analysis simualted data (synthetic data)
sim_path=pathlib.PureWindowsPath(r"G:\My Drive\PhD studies\PhD Project\TcellProject\simulation\dataForPlotting").as_posix()