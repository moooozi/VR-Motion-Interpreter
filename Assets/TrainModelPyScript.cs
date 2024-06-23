using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using System;
using System.IO;
using TMPro;
using System.Linq;

public class TrainModelPyScript : MonoBehaviour
{
    string[] potentialPythonPaths = new string[] { @"./Interpreter/python.exe", "python3.8", "python" };
    string pythonPath = "";
    TextMeshProUGUI consoleOutput;
     
    void Start() {

        consoleOutput = GetComponentInChildren<TextMeshProUGUI>();
        foreach (string path in potentialPythonPaths) {
            try {
                ProcessStartInfo testStart = new ProcessStartInfo();
                testStart.FileName = path;
                Process testProcess = Process.Start(testStart);
                pythonPath = path;
                break; // If the path is valid, break the loop
            } catch (Exception) {
                continue; // If the path is not valid, continue to the next one
            }
        }

        if (string.IsNullOrEmpty(pythonPath)) {
            UnityEngine.Debug.Log("No valid Python interpreter found.");
            return;
        }
    }


    public void RunPythonScript() {
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = pythonPath; // Update with your Python path
        start.Arguments = "train_model.py"; // Update with your script path and arguments
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        Process process = Process.Start(start);
        
        using (StreamReader reader = process.StandardOutput) {
            string result;
            while ((result = reader.ReadLine()) != null) {
                UnityEngine.Debug.Log(result);

                // Add the new line to the console output
                consoleOutput.text += result + "\n";

                // Split the console output into lines
                string[] lines = consoleOutput.text.Split('\n');

                // If there are more than 7 lines, remove the oldest line
                if (lines.Length > 7) {
                    consoleOutput.text = string.Join("\n", lines.Skip(1));
                }
            }
        }
    }
}
