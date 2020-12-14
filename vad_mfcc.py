from bob.bio.spear.preprocessor import Energy_2Gauss, Mod_4Hz, Energy_Thr, External
from bob.bio.spear.database import AudioBioFile
import bob.ap
import numpy as np

def extract_coef(directory, FileName):
    """ choose the VAD you want from the following options: Energy_2Gauss, Mod_4Hz, Energy_Thr, External, as told by the name External is a predefined class ready to host your preferable VAD algorithm if the available option does not meet your needs """
    """ PAY ATTENTION to the default parameter values of the VAD you're using, and make sure they are compatible with the parameters of the feature extractor"""
    vad = Energy_Thr()
    # specify the directory of the audio file, the extension type and the File name
    #directory = '/home/emna/Desktop/DAIC WOZ # EMNA/done/303_P/split/Participant' # this should be only the directory path and not the filename
    """directory = '/home/akomaty/Documents/Projets/Co-encadrement_Emna_Rejaibi/Audio_files/301_P/split/Participant'"""
    extension = '.wav'
    #FileName = '303_AUDIO_0' # Put your filename here
    # put your file in a format compatible with bob
    myfile = AudioBioFile('', FileName, FileName)
    # read the data from your audio file: fs is the sampling frequency and audio signal
    fs, audio_signal = vad.read_original_data(myfile, directory, extension)

    # apply the VAD algorithm to the audio signal, the result will be a set of labels per frame (labels/frame)
    rate, data, labels = vad((fs, audio_signal))
    
    #print(rate) # rate is Hz
    
    """ The LFCC and MFCC coefficients can be extracted from a audio signal by using bob.ap.Ceps. To do so, several parameters can be chosen by the user. Typically, these are chosen in a configuration file. The following values are the default ones:"""
    win_length_ms = 60 # The window length of the cepstral analysis in milliseconds
    win_shift_ms = 40 # The window shift of the cepstral analysis in milliseconds
    n_filters = 24 # The number of filter bands
    n_ceps = 19 # The number of cepstral coefficients
    f_min = 0. # The minimal frequency of the filter bank
    f_max = 8000. # The maximal frequency of the filter bank
    delta_win = 2 # The integer delta value used for computing the first and second order derivatives
    pre_emphasis_coef = 1.0 # The coefficient used for the pre-emphasis
    dct_norm = True # A factor by which the cepstral coefficients are multiplied
    mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale
    
    """ Once the parameters are chosen, bob.ap.Ceps can be called as follows: """
    ceps_mfcc = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    ceps_mfcc.dct_norm = True
    ceps_mfcc.mel_scale = True
    ceps_mfcc.with_energy = True
    ceps_mfcc.with_delta = True
    ceps_mfcc.with_delta_delta = True
    
    data_mfcc = np.cast['float'](data) # vector should be in **float**
    mfcc = ceps_mfcc(data_mfcc)
 
    ceps_lfcc = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    ceps_lfcc.dct_norm = True
    ceps_lfcc.mel_scale = False
    ceps_lfcc.with_energy = True
    ceps_lfcc.with_delta = True
    ceps_lfcc.with_delta_delta = True
    
    data_lfcc = np.cast['float'](data) # vector should be in **float**
    lfcc = ceps_lfcc(data_lfcc)
    
    matrix_coef = np.concatenate((mfcc, lfcc), axis=1) # concatenate both matrix: mfcc and lfcc
    
    return matrix_coef