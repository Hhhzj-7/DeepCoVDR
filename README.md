# DeepCoVDR
DeepCoVDR: Deep transfer learning with graph transformer and cross-attention for predicting COVID-19 drug response


## Data
'GDSC_data/Cell_line_RMA_proc_basalExp.txt' : Cell line data of cancer dataset. Because the file size exceeds the limit, we did not upload it to github. You can download it from [data](https://drive.google.com/file/d/1vvBfkuWJaiBVOQz9sRxe4lfTzgTj4iXE/view?usp=share_link).


'GDSC_data /smile_inchi.csv': Smile data of cancer dataset


'GDSC_data /GDSC2_fitted_dose_response_25Feb20.xlsx': Drug response information of cancer dataset


'SarsCov2_data/Veroe6.xlsx': Vero E6 data for SARS-CoV dataset.


'SarsCov2_data/fda.csv': FDA-approved drugs from https://www.probes-drugs.org/compoundsets .


'SarsCov2_data/sars_ic50.tsv': Drug response information of SARS-CoV dataset.


## Environment
`You can create a conda environment for DeepCoVDR by ‘conda env create -f environment.yml‘.`


## Train and test
`python train.py`
