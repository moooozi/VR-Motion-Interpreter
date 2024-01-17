using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.UIElements;
using TMPro;
using System.Text;
using System.IO;
using UnityEngine.PlayerLoop;
using System;
using System.Globalization;
using UnityEngine.InputSystem;


public class MotionDebugging : MonoBehaviour
{

    [SerializeField] GameObject LeftHand;
    [SerializeField] GameObject RightHand;
    [SerializeField] GameObject Head;

    [SerializeField] GameObject DebugScreen;
    [SerializeField] GameObject StepFeedbackScreen;
    [SerializeField] GameObject CameraOffset;

    private bool recordData = false;
    private float startTime;

    private Vector3 currentLHandPos = Vector3.zero;
    private Vector3 currentRHandPos = Vector3.zero;
    private Vector3 currentHeadPos = Vector3.zero;

    private Vector3 deltaLHandPos = Vector3.zero;
    private Vector3 deltaRHandPos = Vector3.zero;
    private Vector3 deltaHeadPos = Vector3.zero;

    private Quaternion deltaLHandRot = Quaternion.identity;
    private Quaternion deltaRHandRot = Quaternion.identity;
    private Quaternion deltaHeadRot = Quaternion.identity;

    private Vector3 previousLHandPos = Vector3.zero;
    private Vector3 previousHeadPos = Vector3.zero;
    private Vector3 previousRHandPos = Vector3.zero;

    private Quaternion previousLHandRot;
    private Quaternion previousRHandRot;
    private Quaternion previousHeadRot;

    private Quaternion currentLHandRot;
    private Quaternion currentRHandRot;
    private Quaternion currentHeadRot;

    private StepFeedback stepFeedback;
    private string step = "";
    TextMeshProUGUI debugScreenText;

    [SerializeField] float resetInterval = 0.05f; // Reset interval in seconds

    private float resetTimer = 0f;

    private StreamWriter csvWriter;

    [SerializeField] InputActionReference startRecordingAction;
    [SerializeField] InputActionReference leftStepAction;
    [SerializeField] InputActionReference rightStepAction;

    private void Awake()
    {
        debugScreenText = DebugScreen.GetComponent<TextMeshProUGUI>();
        startRecordingAction.action.started += toggleRecording;
        leftStepAction.action.started += registerLeftStep;
        rightStepAction.action.started += registerRightStep;
        stepFeedback = StepFeedbackScreen.GetComponent<StepFeedback>();
    }

    void Update()
    {
        UpdateData();
    }

    /// <summary>
    /// Writes the data to a CSV file.
    /// </summary>
    private void WriteDataToCSV(bool init = false)
    {
        string csvData;

        if (init)
        {
            csvData = string.Join(",", new string[]
            {
                "uptime",
                "step",
                "dPosLHX", "dPosLHY", "dPosLHZ",
                "dRotLHX", "dRotLHY", "dRotLHZ",
                "dPosHeadX", "dPosHeadY", "dPosHeadZ",
                "dRotHeadX", "dRotHeadY", "dRotHeadZ",
                "dPosRHX", "dPosRHY", "dPosRHZ",
                "dRotRHX", "dRotRHY", "dRotRHZ",
                "cPosLHX", "cPosLHY", "cPosLHZ",
                "cRotLHX", "cRotLHY", "cRotLHZ",
                "cPosHeadX", "cPosHeadY", "cPosHeadZ",
                "cRotHeadX", "cRotHeadY", "cRotHeadZ",
                "cPosRHX", "cPosRHY", "cPosRHZ",
                "cRotRHX", "cRotRHY", "cRotRHZ"
            });

            csvWriter.WriteLine(csvData);
            return;
        }
        csvData = string.Join(",", new string[]
        {
            FormatFloat(startTime),
            step,
            FormatFloat(deltaLHandPos[0]), FormatFloat(deltaLHandPos[1]), FormatFloat(deltaLHandPos[2]),
            FormatFloat(deltaLHandRot[0]), FormatFloat(deltaLHandRot[1]), FormatFloat(deltaLHandRot[2]),
            FormatFloat(deltaHeadPos[0]), FormatFloat(deltaHeadPos[1]), FormatFloat(deltaHeadPos[2]),
            FormatFloat(deltaHeadRot[0]), FormatFloat(deltaHeadRot[1]), FormatFloat(deltaHeadRot[2]),
            FormatFloat(deltaRHandPos[0]), FormatFloat(deltaRHandPos[1]), FormatFloat(deltaRHandPos[2]),
            FormatFloat(deltaRHandRot[0]), FormatFloat(deltaRHandRot[1]), FormatFloat(deltaRHandRot[2]),
            FormatFloat(currentLHandPos[0]), FormatFloat(currentLHandPos[1]), FormatFloat(currentLHandPos[2]),
            FormatFloat(currentLHandRot[0]), FormatFloat(currentLHandRot[1]), FormatFloat(currentLHandRot[2]),
            FormatFloat(currentHeadPos[0]), FormatFloat(currentHeadPos[1]), FormatFloat(currentHeadPos[2]),
            FormatFloat(currentHeadRot[0]), FormatFloat(currentHeadRot[1]), FormatFloat(currentHeadRot[2]),
            FormatFloat(currentRHandPos[0]), FormatFloat(currentRHandPos[1]), FormatFloat(currentRHandPos[2]),
            FormatFloat(currentRHandRot[0]), FormatFloat(currentRHandRot[1]), FormatFloat(currentRHandRot[2])
        });
        print(csvData);
        csvWriter.WriteLine(csvData);
    }

    private string FormatFloat(float value)
    {
        return value.ToString("0.00000", CultureInfo.InvariantCulture);
    }


    void toggleRecording(InputAction.CallbackContext context)
    {
        recordData = !recordData;
        if (!recordData){
            return;
        } 
        // If not recording, start recording
        startTime = 0f;

        string timestamp = DateTime.Now.ToString("yyyyMMddHH-mmss");
        string csvFilePath = $"{timestamp}.csv";
        csvWriter = new StreamWriter(csvFilePath, false);
        WriteDataToCSV(init:true);
    }

    void registerLeftStep(InputAction.CallbackContext context)
    {
         step = "left";
         stepFeedback.leftStepped = true;
    }
    void registerRightStep(InputAction.CallbackContext context)
    {
        step = "right";
        stepFeedback.rightStepped = true;
    }
    void UpdateData()
    {
        startTime += Time.deltaTime;
        currentLHandPos = LeftHand.transform.position;
        deltaLHandPos += currentLHandPos - previousLHandPos;
        currentHeadPos = Head.transform.position;
        deltaHeadPos += currentHeadPos - previousHeadPos;

        currentRHandPos = RightHand.transform.position;
        deltaRHandPos += currentRHandPos - previousRHandPos;

        currentLHandRot = LeftHand.transform.rotation;
        deltaLHandRot = Quaternion.Inverse(previousLHandRot) * currentLHandRot;
        currentHeadRot = Head.transform.rotation;
        deltaHeadRot = Quaternion.Inverse(previousHeadRot) * currentHeadRot;
        currentRHandRot = RightHand.transform.rotation;
        deltaRHandRot = Quaternion.Inverse(previousRHandRot) * currentRHandRot;


        resetTimer += Time.deltaTime;
        if (resetTimer >= resetInterval)
        {
            if (recordData) WriteDataToCSV();
            deltaLHandPos = Vector3.zero;
            deltaHeadPos = Vector3.zero;
            deltaRHandPos = Vector3.zero;

            deltaLHandRot = Quaternion.identity;
            deltaHeadRot = Quaternion.identity;
            deltaRHandRot = Quaternion.identity;

            resetTimer = 0f;

        }

        StringBuilder debugScreenTextBuilder = new StringBuilder();
        debugScreenTextBuilder.Append("Recording: ");
        debugScreenTextBuilder.Append(recordData.ToString());
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dPosLH: ");
        debugScreenTextBuilder.Append(deltaLHandPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dRotLH: ");
        debugScreenTextBuilder.Append(deltaLHandRot.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dPosHead: ");
        debugScreenTextBuilder.Append(deltaHeadPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dRotHead: ");
        debugScreenTextBuilder.Append(deltaHeadRot.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dPosRH: ");
        debugScreenTextBuilder.Append(deltaRHandPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dRotRH: ");
        debugScreenTextBuilder.Append(deltaRHandRot.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        // Add current positions
        debugScreenTextBuilder.AppendLine();
        
        debugScreenTextBuilder.Append("cPosLH: ");
        debugScreenTextBuilder.Append(currentLHandPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("cPosHead: ");
        debugScreenTextBuilder.Append(currentHeadPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("cPosRH: ");
        debugScreenTextBuilder.Append(currentRHandPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenText.text = debugScreenTextBuilder.ToString();

        previousLHandPos = currentLHandPos;
        previousHeadPos = currentHeadPos;
        previousRHandPos = currentRHandPos;

        previousLHandRot = currentLHandRot;
        previousHeadRot = currentHeadRot;
        previousRHandRot = currentRHandRot;

        step = "";
    }
}