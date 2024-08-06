# VR Step Recognition

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation for Windows](#installation-for-windows)
- [Usage](#usage)
  - [Record Motion and Step Data](#record-motion-and-step-data)
  - [Training Step Recognition Model](#training-step-recognition-model)
- [License](#license)

## Overview

VR Step Recognition is a project aimed at recognizing and analyzing steps in a virtual reality environment. This project leverages machine learning algorithms to accurately detect and interpret user movements.

## Features

- Real-time motion-based step detection using a pre-trained model.
- Record steps and motion timeline and export it as a CSV file.
- Train machine learning models using timelines (CSV files).
- Replay timeline (CSV file) in Unity.

## Prerequisites

- Unity Editor v2023.2.20f1
- Python 3.8.x (for training models)
- [Winget](https://github.com/microsoft/winget-cli) (optional, for installing Python)

## Installation for Windows

### Unity Setup

1. Clone the project.
2. Open the project with [**Unity Editor v2023.2.20f1**](https://unity.com/releases/editor/whats-new/2023.2.20#installs).

### Python Setup (Optional, for training models)

3. Install a **Python interpreter** that is **supported by PyTorch**:

   - Download and install Python 3.8.x from the [official Python website](https://www.python.org/downloads/release/python-3819/).
   - Or install using **Winget**:

     ```powershell
     winget install Python.Python.3.8
     ```

4. **Create a virtual environment**:
   - Open PowerShell from the project root directory and run the following:

     ```powershell
     python3.8 -m venv .venv
     .\.venv\Scripts\activate 
     ```

   - Or create `venv` inside VSCode. VSCode will ensure the `venv` is activated when the project is open.

5. **Install the required Python dependencies** from `requirements.txt`:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Inside Unity, you can create a new record (CSV), play the last record, and test the selected step detection model.

### Record Motion and Step Data

1. Start a new record using the Game UI in Unity.
   - When you start recording, the motion data of the headset and both controllers will be recorded until you stop recording.
   - To record your steps in sync with the motion data, touch the Joystick of the left/right controller as soon as your left/right foot touches the ground. You should see the grey screen turn blue/red to provide feedback, so you know that your step was registered.

**Note:** When you record a timeline, the CSV will be saved in `Assets\MLTrainingData`.

**Important:** By default, all CSV files inside `Assets\MLTrainingData` will be used to train and test the next step recognition model. Therefore, it is recommended to delete low-quality CSVs from prior recordings.

**Tips for High-Quality Recordings:**

- Stop the recording and start a new one from time to time, so it's easier to delete bad recordings when you register or miss a step by accident.
- Stand still occasionally without performing steps.
- Try stepping with different speeds.
- Time your step registering well, ideally as soon as your feet touch the ground.

**Technical Detail:** By default, the first and last second of the recording is removed to filter out the noise caused by trying to find and click the `Start/Stop Recording` button.

### Training Step Recognition Model

**Preparation:** The CSVs inside `Assets\MLTrainingData` are split into two types:

- **Training data**: All recorded CSVs by default.
- **Evaluation data**: All CSVs that start with `eval_`.

Rename some of your CSV files to start with `eval_`, such as `eval_1`, `eval_23`.

Make sure that **at least one** CSV filename starts with `eval_` and **another one that doesn't**.

Run the Python script `train_model.py` from the project root directory:

```powershell
python train_model.py
```

**Tips:**

- A good ratio of training/evaluation data split is 80/20.
- Use the highest quality CSVs for evaluation.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for more information.
