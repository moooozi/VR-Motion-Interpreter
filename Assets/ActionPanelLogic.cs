using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActionPanelLogic : MonoBehaviour
{
    [SerializeField] GameObject btnToggleRecording;
    [SerializeField] GameObject btnToggleRecognition;

    private TMPro.TextMeshProUGUI btnToggleRecordingText;
    private TMPro.TextMeshProUGUI btnToggleRecognitionText;

    [SerializeField] private TMPro.TextMeshProUGUI recordingHintText;

    private MotionDebugging motionDebugging;

    private bool isRecording = false;
    private bool isRecognition = false;

    void Start()
    {
        btnToggleRecognitionText = btnToggleRecognition.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        btnToggleRecordingText = btnToggleRecording.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        motionDebugging = GameObject.FindGameObjectWithTag("XROrigin").GetComponent<MotionDebugging>();

        btnToggleRecordingText.text = "Start Recording";
        btnToggleRecognitionText.text = "Start Recognition";
    }

    public void ToggleRecording()
    {
        isRecording = motionDebugging.toggleRecording();

        if (isRecording)
        {
            if (isRecognition)
            {
                ToggleRecognition();
            }
            btnToggleRecordingText.text = "Stop Recording";
            recordingHintText.enabled = true;
        }
        else
        {
            btnToggleRecordingText.text = "Start Recording";
            recordingHintText.enabled = false;
        }
    }

    public void ToggleRecognition()
    {
        isRecognition = motionDebugging.toggleRecognition();
        if (isRecognition)
        {
            if (isRecording)
            {
                ToggleRecording();
            }
            btnToggleRecognitionText.text = "Stop Recognition";
        }
        else
        {
            btnToggleRecognitionText.text = "Start Recognition";
        }
    }


}
