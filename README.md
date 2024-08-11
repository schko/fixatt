# Code package for Fixating on Attention

Environment: Setup using https://huggingface.co/docs/timm/installation#from-source (see **NOTE 1**).

In order, the major steps required to reproduce the paper are:
- Data (sample data sent with package, see **NOTE 2**)
- Preprocessing
- Training
- Analysis

These are located within their respective folders with a documented implementation (in `.ipynb`s) and description of relevant functions (in `.py`s). You will first need to define the correct configurations in the `config.ini` file.

---

**NOTE 1**: 
Building *timm* from source lets you make changes to the code base. To train the baseline ViT model with the FAX loss, replace the content within the file `pytorch-image-models/timm/models/vision_transformer.py` with the content sourced from `vision_transformer_modified_for_fax_loss.py`. This alternative code file is available within the `fixatt/training` subfolder.

**NOTE 2**:
To access the full dataset, download the fix_att folder from Google Drive into the local directory: https://drive.google.com/drive/folders/1Z-GEwl_e2XlJwsMhK6aUhNubcuYs0aRd?usp=sharing. The resulting folder with preprocessed data from both datasets files will be in the format:
- data
    - VR (~60 GB)
        - Subject_#-Session_#
            - s_ms (where s = seconds of motor action onset, ms = milliseconds percision)
                - avg_trial_data.p (pickle file containing premotor averaged variable, data_dict)
    - dreyeve
        - run_#
            - frame# (frame number, in the original study frames, corresponding to the motor action onset)
                - avg_trial_data.p (pickle file containing premotor averaged variable, data_dict)