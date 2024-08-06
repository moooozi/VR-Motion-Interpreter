# VR Step Recognition

## Overview

VR Step Recognition is a project aimed at recognizing and analyzing steps in a virtual reality environment. This project leverages advanced machine learning algorithms to accurately detect and interpret user movements.

## Features

- Real-time motion-based step detection using pre-trained model.
- Record steps and motion timeline and export it as CSV file.
- Train machine learning models using one or many pre-recorded timelines (CSV files).
- Replay timeline (CSV file) in Unity.

## Installation for Windows

1. Clone the project
2. Open the project with [**Unity Editor v2023.2.20f1**](https://unity.com/releases/editor/whats-new/2023.2.20#installs)

### (Optional) If you need to train models, follow the additional steps below:

3. **Install a **Python interpreter** that is **supported by PyTorch**:
   - Download and install Python 3.8.x from the [official Python website](https://www.python.org/downloads/release/python-3819/).

   - or install using **Winget**:

        ```powershell
        winget install Python.Python.3.8
        ```

4. **Create a virtual environment**:

   - Open Powershell from the project root directory and run the following:

        ```powershell
        python3.8 -m venv .venv
        .\.venv\Scripts\activate 
        ```

   - Or create `venv` inside VSCode. VSCode will make sure the `venv` is activated when the project is open.

5. **Install the required Python dependencies** from `requirements.txt`:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Inside Unity, you can create a new record (CSV), play last record and test the selected step detection model.

### Record motion and step data

Start a new record using the Game UI in Unity.

- When you start recording, the motion data of the headset and both controllers will be recorded until you stop recording.
- To record your steps in sync with the motion data, touch the Joystick of the left/right controller as soon as your left/right foot touches the ground. You should see the grey screen turn blue/red to provide you with feedback, so you know that your step was registered.

**Note:** When you record data, the CSV will be saved in `Assets\MLTrainingData`

**Important:** By default, all CSV files inside `Assets\MLTrainingData` will be used to train and test the next step recognition model. Therefore it is recommanded to delete low-quality CSVs.

**Tips for high quality recordings:**

- Stop the recording and start a new one from time to time, so its easier to delete bad records when you register or miss a step by accident.

- Stand still occationally without performing steps.

- Try stepping with different speeds.

- Time your step registering well, ideally as soon as your feet touch the ground.

**Technical detail:** By default, the first and last secord of the recording is removed to filter out the noise caused by trying to find and click the `Start/Stop Recording` button.

### Training step recognition model

**Preparation:** The CSVs inside `Assets\MLTrainingData` are split into two types.

- **Training data**: All recorded CSVs by default.
- **Evaluation data**: All CSVs that start with `eval_`

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
