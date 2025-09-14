# Arabidopsis Root Cortex Cell Tracking

![Arabidopsis root cortex cell tracking overall architecture](assets/overview.png?raw=true)

This repository presents a novel method for tracking Arabidopsis root cortex cells using a tracking-by-detection approach.

## Overview

### Detection Phase
We employ a semantic segmentation approach with a U-Net backbone to detect cells and classify dividing and non-dividing cells. Pre-trained weights are provided. Our GUI system generates files compatible with ImageJ/Fiji [TrackMate](https://imagej.net/plugins/trackmate/), allowing for easy manual refinement.

### Tracking Phase
We introduce an accurate tracking method based on a Genetic Algorithm (GA) and K-means clustering. Our coarse-to-fine strategy begins with cell file-level tracking, where GA selects an optimal projection plane. K-means then clusters the projected cells into eight groups, each representing a distinct line of cells. Our GUI provides tools for correcting clustering errors, and the final tracking results can be visualized in TrackMate.

With precise detection and clustering, Arabidopsis root nuclei can be accurately tracked. To our knowledge, this is the first successful attempt to address a long-standing challenge in time-lapse microscopy of root meristems by providing an accurate tracking method for Arabidopsis root nuclei.

For more details, refer to our paper: [[`Paper`](https://academic.oup.com/pcp/article/64/11/1262/7323573)].

## Installation

We provide a GUI system for this method. We tested our code on **Python 3.8** with **Torch 2.3.0** on **Ubuntu20.0.4**. We recommend using **Anaconda** to set up an environment and install dependencies:

### Steps:

```bash
conda create -n Arabidopsis python=3.8
conda activate Arabidopsis
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/JerrySongCST/Arabidopsis_root_cortex_cell_tracking.git && cd Arabidopsis_root_cortex_cell_tracking
pip install -e .
```
For ubuntu users:
```bash
sudo apt-get install libxcb-xinerama0
```

### Downloading Checkpoints for Detection Algorithm

Users need to download model checkpoints [here](https://drive.google.com/drive/folders/1XdNGD-tufMjMFptxqqRKve0RJdr8RXB9?usp=sharing). Place the checkpoint files in the appropriate folder as shown:

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


### Cell Detection 

1. Downloading Pre-trained Model Checkpoints for Detection Algorithm.

2. Click **"Cell Detection"**:
   
   ![Detection Button](assets/detection_ui.jpg?raw=true)


3. Click **"Choose"** and select the folder containing the `.tif` file (not the file itself!). Ensure that the .tif file is inside the selected folder. Do not include any .xml file that shares the same name as the .tif file in the selected folder


   ![Choose Button](assets/choose1.jpg?raw=true)


4. Select either:

   - **2D Model** – Performs slice-by-slice detection using a trained 2D U-Net.
   
   - **3D Model** – Processes the entire volume per frame using a trained 3D U-Net (requires significantly more GPU memory).
   
5. Click **"Automatic Detect"** to start. **GPU acceleration is strongly recommended.**


   ![Detect Button](assets/detect.jpg?raw=true)

6. Once completed, the GUI generates an XML file that can be loaded into [TrackMate](https://imagej.net/plugins/trackmate/). The name is same as tif image file name.

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


8. Click **"Link Lines"** and **"Track Cells"** to finalize tracking.


   ![Link Lines](assets/link_lines.jpg?raw=true)


   ![Track Cells](assets/cell_track.png?raw=true)

9. Click **"Save XML"** (don't forget the `.xml` extension). Results can be loaded into [TrackMate](https://imagej.net/plugins/trackmate/).


   ![Save XML](assets/save_xml.jpg?raw=true)


### Additional Tools

- **"Reload XML"**: This function reloads the detection XML file after modifications have been made. Use this feature when nuclei have been added or removed in the TrackMate XML file after performing clustering on the cell file.


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

@article{song2025accurate,
  title={Accurate Tracking of Arabidopsis Root Cortex Cell Nuclei in 3D Time-Lapse Microscopy Images Based on Genetic Algorithm},
  author={Song, Yu and Goh, Tatsuaki and Li, Yinhao and Dong, Jiahua and Miyashima, Shunsuke and Iwamoto, Yutaro and Kondo, Yohei and Nakajima, Keiji and Chen, Yen-wei},
  journal={arXiv preprint arXiv:2504.12676},
  year={2025}
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

