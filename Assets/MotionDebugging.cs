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

    private SensorData leftHand = new SensorData();
    private SensorData rightHand = new SensorData();
    private SensorData head = new SensorData();

    private StepFeedback stepFeedback;
    private int step = 0;
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
            step.ToString(),
            FormatFloat(leftHand.DeltaPos.x), FormatFloat(leftHand.DeltaPos.y), FormatFloat(leftHand.DeltaPos.z),
            FormatFloat(leftHand.DeltaRot.x), FormatFloat(leftHand.DeltaRot.y), FormatFloat(leftHand.DeltaRot.z),
            FormatFloat(head.DeltaPos.x), FormatFloat(head.DeltaPos.y), FormatFloat(head.DeltaPos.z),
            FormatFloat(head.DeltaRot.x), FormatFloat(head.DeltaRot.y), FormatFloat(head.DeltaRot.z),
            FormatFloat(rightHand.DeltaPos.x), FormatFloat(rightHand.DeltaPos.y), FormatFloat(rightHand.DeltaPos.z),
            FormatFloat(rightHand.DeltaRot.x), FormatFloat(rightHand.DeltaRot.y), FormatFloat(rightHand.DeltaRot.z),
            FormatFloat(leftHand.CurrentPos.x), FormatFloat(leftHand.CurrentPos.y), FormatFloat(leftHand.CurrentPos.z),
            FormatFloat(leftHand.CurrentRot.x), FormatFloat(leftHand.CurrentRot.y), FormatFloat(leftHand.CurrentRot.z),
            FormatFloat(head.CurrentPos.x), FormatFloat(head.CurrentPos.y), FormatFloat(head.CurrentPos.z),
            FormatFloat(head.CurrentRot.x), FormatFloat(head.CurrentRot.y), FormatFloat(head.CurrentRot.z),
            FormatFloat(rightHand.CurrentPos.x), FormatFloat(rightHand.CurrentPos.y), FormatFloat(rightHand.CurrentPos.z),
            FormatFloat(rightHand.CurrentRot.x), FormatFloat(rightHand.CurrentRot.y), FormatFloat(rightHand.CurrentRot.z)
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
         step = 1;
         stepFeedback.leftStepped = true;
    }
    void registerRightStep(InputAction.CallbackContext context)
    {
        step = 2;
        stepFeedback.rightStepped = true;
    }
    void UpdateData()
    {
        startTime += Time.deltaTime;
        
        leftHand.CurrentPos = LeftHand.transform.position;
        leftHand.DeltaPos += leftHand.CurrentPos - leftHand.PreviousPos;
        
        head.CurrentPos = Head.transform.position;
        head.DeltaPos += head.CurrentPos - head.PreviousPos;
        
        rightHand.CurrentPos = RightHand.transform.position;
        rightHand.DeltaPos += rightHand.CurrentPos - rightHand.PreviousPos;
        
        leftHand.CurrentRot = LeftHand.transform.rotation;
        leftHand.DeltaRot = Quaternion.Inverse(leftHand.PreviousRot) * leftHand.CurrentRot;
        
        head.CurrentRot = Head.transform.rotation;
        head.DeltaRot = Quaternion.Inverse(head.PreviousRot) * head.CurrentRot;
        
        rightHand.CurrentRot = RightHand.transform.rotation;
        rightHand.DeltaRot = Quaternion.Inverse(rightHand.PreviousRot) * rightHand.CurrentRot;
        
        resetTimer += Time.deltaTime;
        if (resetTimer >= resetInterval)
        {
            if (recordData) WriteDataToCSV();
            leftHand.DeltaPos = Vector3.zero;
            head.DeltaPos = Vector3.zero;
            rightHand.DeltaPos = Vector3.zero;
        
            leftHand.DeltaRot = Quaternion.identity;
            head.DeltaRot = Quaternion.identity;
            rightHand.DeltaRot = Quaternion.identity;
        
            resetTimer = 0f;
        }
        StringBuilder debugScreenTextBuilder = new StringBuilder();
        debugScreenTextBuilder.Append("Recording: ");
        debugScreenTextBuilder.Append(recordData.ToString());
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dPosLH: ");
        debugScreenTextBuilder.Append(leftHand.DeltaPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dRotLH: ");
        debugScreenTextBuilder.Append(leftHand.DeltaRot.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dPosHead: ");
        debugScreenTextBuilder.Append(head.DeltaPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dRotHead: ");
        debugScreenTextBuilder.Append(head.DeltaRot.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dPosRH: ");
        debugScreenTextBuilder.Append(rightHand.DeltaPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("dRotRH: ");
        debugScreenTextBuilder.Append(rightHand.DeltaRot.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        // Add current positions
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("cPosLH: ");
        debugScreenTextBuilder.Append(leftHand.CurrentPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("cPosHead: ");
        debugScreenTextBuilder.Append(head.CurrentPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("cPosRH: ");
        debugScreenTextBuilder.Append(rightHand.CurrentPos.ToString("F3"));
        debugScreenTextBuilder.AppendLine();

        debugScreenText.text = debugScreenTextBuilder.ToString();

        leftHand.PreviousPos = leftHand.CurrentPos;
        head.PreviousPos = head.CurrentPos;
        rightHand.PreviousPos = rightHand.CurrentPos;

        leftHand.PreviousRot = leftHand.CurrentRot;
        head.PreviousRot = head.CurrentRot;
        rightHand.PreviousRot = rightHand.CurrentRot;

        step = 0;
    }
}


public class SensorData
{
    public Vector3 CurrentPos { get; set; } = Vector3.zero;
    public Vector3 DeltaPos { get; set; } = Vector3.zero;
    public Vector3 PreviousPos { get; set; } = Vector3.zero;

    public Quaternion CurrentRot { get; set; } = Quaternion.identity;
    public Quaternion DeltaRot { get; set; } = Quaternion.identity;
    public Quaternion PreviousRot { get; set; } = Quaternion.identity;
}