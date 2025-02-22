# Arabidopsis Root Cortex Cell Tracking

![Arabidopsis root cortex cell tracking overall architecture](assets/overview.png?raw=true)

This repository presents a novel method for tracking Arabidopsis root cortex cells using a tracking-by-detection approach.

## Overview

### Detection Phase
We employ a semantic segmentation approach with a U-Net backbone to detect cells and classify dividing and non-dividing cells. Pre-trained weights are provided. Our GUI system generates files compatible with ImageJ/Fiji TrackMate, allowing for easy manual refinement.

### Tracking Phase
We introduce an accurate tracking method based on a Genetic Algorithm (GA) and K-means clustering. Our coarse-to-fine strategy begins with line-level tracking of cell nuclei, where GA selects an optimal projection plane. K-means then clusters the projected cells into eight groups, each representing a distinct line of cells. Our GUI provides tools for correcting clustering errors, and the final tracking results can be visualized in TrackMate.

With precise detection and clustering, Arabidopsis root nuclei can be accurately tracked. To our knowledge, this is the first successful attempt to address a long-standing challenge in time-lapse microscopy of root meristems by providing an accurate tracking method for Arabidopsis root nuclei.

For more details, refer to our paper: [[`Paper`](https://academic.oup.com/pcp/article/64/11/1262/7323573)].

## Installation

We provide a GUI system for this method. If you do not wish to install Python and PyTorch manually, you can download the packaged ZIP file [here](https://drive.google.com/file/d/15m6AjEMTTf5cfnT3oOeC8sA0WWa-XDQE/view?usp=sharing) and run the executable (`.exe`). Note: This is only available for Windows users.

For manual installation, we tested our code on **Python 3.10** with **Torch 2.3.0**. We recommend using **Anaconda** to set up an environment and install dependencies:

### Steps:

```bash
conda create -n Arabidopsis python=3.10
conda activate Arabidopsis
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/JerrySongCST/Arabidopsis_root_cortex_cell_tracking.git && cd Arabidopsis_root_cortex_cell_tracking
pip install -e .
```

### Downloading Checkpoints for Detection Algorithm

Both ZIP file users and manual installation users need to download model checkpoints [here](https://drive.google.com/drive/folders/1XdNGD-tufMjMFptxqqRKve0RJdr8RXB9?usp=sharing). Place the checkpoint files in the appropriate folder as shown:

![Checkpoint placement](assets/pth.png?raw=true)

For cell detection, we strongly recommend using a **GPU** for faster processing.

## Dataset

A **4D microscopy system** developed by Tatsuaki Goh et al. at **Nara Institute of Science and Technology (NAIST)** captures live imaging of Arabidopsis roots (3D + Time). 

The dataset is private, but we provide **10 sample frames** for testing: [Download here](https://drive.google.com/drive/folders/1l8Ij9N3ODNBB29kc-vXhjcDnrSU2eUdR?usp=drive_link). Ground-truth cell locations and types are included for tracking validation.

## Getting Started

Run the GUI using:
```bash
python launch.py
```

Alternatively, double-click **"launch.exe"**.

### Cell Detection

1. Click **"Cell Detection"**:

   
   ![Detection Button](assets/detection_ui.jpg?raw=true)


2. Click **"Choose"** and select the folder containing the `.tif` file (not the file itself!).


   ![Choose Button](assets/choose1.jpg?raw=true)


3. Select either:

   - **2D Model** (performs slice-by-slice detection using a trained 2D U-Net)
   
   - **3D Model** (processes the entire volume per frame using a trained 3D U-Net)
   
4. Click **"Automatic Detect"** to start. **GPU acceleration is strongly recommended.**


   ![Detect Button](assets/detect.jpg?raw=true)

5. Once completed, the GUI generates an XML file that can be loaded into TrackMate. The name is same as tif image file name.

### Tracking

1. Click **"Cell Tracking"**:


   ![Tracking Button](assets/tracking_ui.jpg?raw=true)


2. Click **"Choose"** and select the folder containing the `.tif` and `.xml` detection files.


   ![Choose Button](assets/choose2.jpg?raw=true)


3. Click **"Automatic Clustering"** to run GA and K-means clustering.


   ![Automatic Clustering](assets/ga_clustering.jpg?raw=true)


4. Click **"Check Clustering"** to visualize results. If clicked before clustering process finished, please click "Refresh" after clustering is finished.


   ![Check Clustering Result](assets/check_clustering.jpg?raw=true)


5. Correct misclassified cells using color selection.


   ![Rectifying Clustering](assets/Rectification.jpg?raw=true)


   ![Rectifying Clustering](assets/Rectification2.jpg?raw=true)


6. Use the **Next** and **Previous** buttons to review all frames.


   ![Frame Navigation](assets/next.jpg?raw=true)


7. Click **"Reset Cell ID"** to finalize lineage order. Then click **"Next Step"**.


   ![Reset Cell ID](assets/reset_id.jpg?raw=true)


8. Click **"Link Lines"**, **"Check Mitotic"**, and **"Track Cells"** to finalize tracking.


   ![Link Lines](assets/link_lines.jpg?raw=true)


   ![Check Mitotic](assets/check_mitotic.jpg?raw=true)


   ![Track Cells](assets/cell_track.jpg?raw=true)

9. Click **"Save XML"** (don't forget the `.xml` extension). Results can be loaded into TrackMate.


   ![Save XML](assets/save_xml.jpg?raw=true)


### Additional Tools

- **"Reload XML"**: Reloads detection XML if changes were made.


   ![Reload XML](assets/reload_xml.jpg?raw=true)


- **"Delete"**: Permanently removes a cell (use with caution).


   ![Delete Cell](assets/delete.jpg?raw=true)


- **Jump to Frame**: Enter a frame number and press **Enter**.


   ![Jump to Frame](assets/jump_frame.jpg?raw=true)


## Citation

If you use our code or dataset, please cite:

```bibtex
@article{goh2023depth,
  title={In-depth quantification of cell division and elongation dynamics at the tip of growing Arabidopsis roots using 4D microscopy, AI-assisted image processing and data sonification},
  author={Goh, Tatsuaki and Song, Yu and Yonekura, Takaaki and others},
  journal={Plant and Cell Physiology},
  volume={64},
  number={11},
  pages={1262--1278},
  year={2023},
  publisher={Oxford University Press UK}
}

@article{song2024dividing,
  title={Dividing and Non-dividing Cell Detection by Segmentation on Arabidopsis Root Images Using Light-weight U-Net},
  author={Song, Yu and Deng, Zeping and Li, Yinhao and others},
  journal={IEICE Proceedings Series},
  volume={81},
  number={S5-3},
  year={2024},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}
```

