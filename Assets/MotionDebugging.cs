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
using Unity.VisualScripting.FullSerializer;
using System.Linq;


public class MotionDebugging : MonoBehaviour
{

    [SerializeField] GameObject LeftHand;
    [SerializeField] GameObject RightHand;
    [SerializeField] GameObject Head;

    [SerializeField] GameObject Sentis;

    [SerializeField] GameObject DebugScreen;
    [SerializeField] GameObject StepFeedbackScreen;
    [SerializeField] GameObject CameraOffset;

    [SerializeField] private bool recordingDataIsOn = false;
    [SerializeField] private bool recognitionIsOn = false;

    private float startTime;

    private SensorData leftHand = new();
    private SensorData rightHand = new();
    private SensorData head = new();

    private MLModelStepRecogize mlModelStepRecogize;

    private float[] recognitionResults = new float[3];

    private StepFeedback stepFeedback;
    private int step = 0;
    TextMeshProUGUI debugScreenText;

    [SerializeField] int recordingFPS = 20; 
    private int finishedEntries = 0;

    private MLSequence mLSequence = new MLSequence(7);

    private List<FullDataFrame> recordedData = new List<FullDataFrame>();
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
        mlModelStepRecogize = Sentis.GetComponent<MLModelStepRecogize>();
    }

    /// <summary>
    /// Writes the data to a CSV file.
    /// </summary>
    private void WriteDataToCSV()
    {
        StringBuilder allLines = new StringBuilder();

        string header = string.Join(",", new string[]
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
        allLines.AppendLine(header);

        // Skip the first and last second of data
        for (int i = recordingFPS; i < recordedData.Count - recordingFPS; i++)
        {
            FullDataFrame frame = recordedData[i];

            string line = string.Join(",", new string[]
            {
                FormatFloat(frame.Uptime),
                frame.Step.ToString(),
                FormatFloat(frame.LeftHand.DeltaPos.x), FormatFloat(frame.LeftHand.DeltaPos.y), FormatFloat(frame.LeftHand.DeltaPos.z),
                FormatFloat(frame.LeftHand.DeltaRot.x), FormatFloat(frame.LeftHand.DeltaRot.y), FormatFloat(frame.LeftHand.DeltaRot.z),
                FormatFloat(frame.Head.DeltaPos.x), FormatFloat(frame.Head.DeltaPos.y), FormatFloat(frame.Head.DeltaPos.z),
                FormatFloat(frame.Head.DeltaRot.x), FormatFloat(frame.Head.DeltaRot.y), FormatFloat(frame.Head.DeltaRot.z),
                FormatFloat(frame.RightHand.DeltaPos.x), FormatFloat(frame.RightHand.DeltaPos.y), FormatFloat(frame.RightHand.DeltaPos.z),
                FormatFloat(frame.RightHand.DeltaRot.x), FormatFloat(frame.RightHand.DeltaRot.y), FormatFloat(frame.RightHand.DeltaRot.z),
                FormatFloat(frame.LeftHand.CurrentPos.x), FormatFloat(frame.LeftHand.CurrentPos.y), FormatFloat(frame.LeftHand.CurrentPos.z),
                FormatFloat(frame.LeftHand.CurrentRot.x), FormatFloat(frame.LeftHand.CurrentRot.y), FormatFloat(frame.LeftHand.CurrentRot.z),
                FormatFloat(frame.Head.CurrentPos.x), FormatFloat(frame.Head.CurrentPos.y), FormatFloat(frame.Head.CurrentPos.z),
                FormatFloat(frame.Head.CurrentRot.x), FormatFloat(frame.Head.CurrentRot.y), FormatFloat(frame.Head.CurrentRot.z),
                FormatFloat(frame.RightHand.CurrentPos.x), FormatFloat(frame.RightHand.CurrentPos.y), FormatFloat(frame.RightHand.CurrentPos.z),
                FormatFloat(frame.RightHand.CurrentRot.x), FormatFloat(frame.RightHand.CurrentRot.y), FormatFloat(frame.RightHand.CurrentRot.z)
            });
            allLines.AppendLine(line);
        }
        recordedData.Clear();
        csvWriter.Write(allLines.ToString());
    }

    private string FormatFloat(float value)
    {
        return value.ToString("0.00000", CultureInfo.InvariantCulture);
    }


    public void toggleRecording(InputAction.CallbackContext context)   {
        recordingDataIsOn = !recordingDataIsOn;
        if (!recordingDataIsOn){
            WriteDataToCSV();
            return;
        } 
        // If not recording, start recording
        startTime = 0f;
        finishedEntries = 0;

        string timestamp = DateTime.Now.ToString("yyyyMMddHH-mmss");
        string csvFilePath = $"Assets/MLTrainingData/{timestamp}.csv";
        csvWriter = new StreamWriter(csvFilePath, false);
    }

    public void toggleRecording() {
        toggleRecording(new InputAction.CallbackContext());
    }

    public void toggleRecognition(InputAction.CallbackContext context)
    {
        recognitionIsOn = !recognitionIsOn;
        
        if (!recognitionIsOn){

            return;
        }
        // Before starting the recognition
        mLSequence.Clear();
        startTime = 0f;
        finishedEntries = 0;
    }

    public void toggleRecognition() {
        toggleRecognition(new InputAction.CallbackContext()); 
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

    void updateRecognition()
    {
        startTime += Time.deltaTime;
        if ((startTime / finishedEntries >= 1.0f / recordingFPS) || finishedEntries == 0) {
            finishedEntries++;
            mLSequence.AddFrame(new MLDataFrame(
                leftHand.DeltaPos, leftHand.DeltaRot, 
                head.DeltaPos, head.DeltaRot, 
                rightHand.DeltaPos, rightHand.DeltaRot, 
                leftHand.CurrentPos - head.CurrentPos, 
                rightHand.CurrentPos - head.CurrentPos));        
        }

        if (!mLSequence.IsFull()) {
            return;
        }
        // to save compute power, only run the model once every 2 frames
        if (mLSequence.oddEven) {
            return;
        }
        // Run the model
        recognitionResults = mlModelStepRecogize.ExecuteModel(mLSequence.GetMLSequence());
        // Find the index of the highest value
        int highestIndex = Array.IndexOf(recognitionResults, recognitionResults.Max());
        
        // Set the step feedback
        stepFeedback.leftStepped = highestIndex == 1;
        stepFeedback.rightStepped = highestIndex == 2;
    }

    void Update()
    {
        leftHand.UpdateSensorData(LeftHand.transform.position, LeftHand.transform.rotation);
        head.UpdateSensorData(Head.transform.position, Head.transform.rotation);
        rightHand.UpdateSensorData(RightHand.transform.position, RightHand.transform.rotation);
    
        if (recordingDataIsOn) {
            startTime += Time.deltaTime;
            if ((startTime / finishedEntries >= 1.0f / recordingFPS) || finishedEntries == 0) {
                finishedEntries++;
                recordedData.Add(new FullDataFrame(startTime, step, new SensorData(leftHand), new SensorData(rightHand), new SensorData(head)));
                step = 0;
            }        
        }
        if (recognitionIsOn)
            updateRecognition();


        StringBuilder debugScreenTextBuilder = new StringBuilder();

        debugScreenTextBuilder.Append("FPS: ");
        debugScreenTextBuilder.Append(1 / Time.deltaTime);
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("Recording: ");
        debugScreenTextBuilder.Append(recordingDataIsOn.ToString());
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("Recognition: ");
        debugScreenTextBuilder.Append(recognitionIsOn.ToString());
        debugScreenTextBuilder.AppendLine();

        debugScreenTextBuilder.Append("Recognition Data: ");
        debugScreenTextBuilder.Append(string.Join(", ", recognitionResults));
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
    
    // Parameterless constructor
    public SensorData() { }
    
    // Copy constructor
    public SensorData(SensorData original)
    {
        CurrentPos = original.CurrentPos;
        DeltaPos = original.DeltaPos;
        PreviousPos = original.PreviousPos;

        CurrentRot = original.CurrentRot;
        DeltaRot = original.DeltaRot;
        PreviousRot = original.PreviousRot;
    }
    public void UpdateSensorData(Vector3 currentPos, Quaternion currentRot)
    {
        PreviousPos = CurrentPos;
        CurrentPos = currentPos;
        DeltaPos = CurrentPos - PreviousPos;

        PreviousRot = CurrentRot;
        CurrentRot = currentRot;
        DeltaRot = Quaternion.Inverse(PreviousRot) * CurrentRot;
    }
}


public class FullDataFrame
{
    public float Uptime { get; set; }
    public int Step { get; set; }

    public SensorData LeftHand { get; set; }
    public SensorData RightHand { get; set; }
    public SensorData Head { get; set; }

    public FullDataFrame(float uptime, int step, SensorData leftHand, SensorData rightHand, SensorData head)
    {
        Uptime = uptime;
        Step = step;
        LeftHand = leftHand;
        RightHand = rightHand;
        Head = head;
    }
}

public class MLDataFrame
{
    public Vector3 LeftHandPos { get; set; }
    public Quaternion LeftHandRot { get; set; }
    
    public Vector3 RightHandPos { get; set; }
    public Quaternion RightHandRot { get; set; }

    public Vector3 HeadPos { get; set; }
    public Quaternion HeadRot { get; set; }

    public Vector3 RelLeftHandPos { get; set; }
    public Vector3 RelRightHandPos { get; set; }


    // Constructor
    public MLDataFrame(Vector3 leftHandPos, Quaternion leftHandRot, Vector3 headPos, Quaternion headRot, Vector3 rightHandPos, Quaternion rightHandRot, Vector3 relLeftHandPos, Vector3 relRightHandPos)
    {
        LeftHandPos = leftHandPos;
        LeftHandRot = leftHandRot;
        RightHandPos = rightHandPos;
        RightHandRot = rightHandRot;
        HeadPos = headPos;
        HeadRot = headRot;
        RelLeftHandPos = relLeftHandPos;
        RelRightHandPos = relRightHandPos;
    }
}

public class MLSequence
{
    private List<MLDataFrame> frames;
    private int maxSize;
    public int frameIndex = 0;
    public bool oddEven = false;

    public MLSequence(int maxSize)
    {
        this.maxSize = maxSize;
        frames = new List<MLDataFrame>();
    }

    public bool IsFull()
    {
        return frames.Count == maxSize;
    }


    public void AddFrame(MLDataFrame frame)
    {
        frames.Add(frame);

        // Ensure that the sequence contains a maximum of maxSize frames
        if (frames.Count > maxSize)
        {
            frames.RemoveAt(0);
        }
        oddEven = !oddEven;
        if (frameIndex < maxSize) frameIndex++;
        else                      frameIndex = 0;
    }

    public List<MLDataFrame> GetFrames()
    {
        return frames;
    }

    public float[][] GetMLSequence()
    {
        // Initialize a new float array to hold the data
        float[][] data = new float[frames.Count][];

        // Iterate over each frame in the sequence
        for (int i = 0; i < frames.Count; i++)
        {
            // Extract the component values from each Vector3 and Quaternion
            data[i] = new float[]
            {
                frames[i].LeftHandPos.x    , frames[i].LeftHandPos.y    , frames[i].LeftHandPos.z,
                frames[i].LeftHandRot.x    , frames[i].LeftHandRot.y    , frames[i].LeftHandRot.z,
                frames[i].RightHandPos.x   , frames[i].RightHandPos.y   , frames[i].RightHandPos.z,
                frames[i].RightHandRot.x   , frames[i].RightHandRot.y   , frames[i].RightHandRot.z,
                frames[i].HeadPos.x        , frames[i].HeadPos.y        , frames[i].HeadPos.z,
                frames[i].HeadRot.x        , frames[i].HeadRot.y        , frames[i].HeadRot.z,
                frames[i].RelLeftHandPos.x , frames[i].RelLeftHandPos.y , frames[i].RelLeftHandPos.z,
                frames[i].RelRightHandPos.x, frames[i].RelRightHandPos.y, frames[i].RelRightHandPos.z
            };
        }
        // Return the prepared data
        return data;
    }
    public void Clear()
    {
        frames.Clear();
        frameIndex = 0;
    }
}