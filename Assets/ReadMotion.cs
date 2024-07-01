using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Globalization;
using Unity.VisualScripting;

public class ReadMotion : MonoBehaviour
{
    [SerializeField] GameObject LeftHand;
    [SerializeField] GameObject RightHand;
    [SerializeField] GameObject Head;

    [SerializeField] GameObject LeftFoot;
    [SerializeField] GameObject RightFoot;

    private Replay replay;

    private bool isPlaying = false;

    Vector3 positionOffset;
    Quaternion rotationOffset;

    // Start is called before the first frame update

    public void Start() {
        // Define position and rotation offsets
         positionOffset = gameObject.transform.position;
         rotationOffset = gameObject.transform.rotation;
    }
    public void PlayCSV(string file_path)
    {
        Debug.Log("Playing CSV file: " + file_path);
        replay = new Replay(file_path);
        isPlaying = true;
        StartCoroutine(ReplayData());
    }

    public void StopCSV()
    {
        StopAllCoroutines();
        isPlaying = false;
    }

    public IEnumerator VisualizeStep(int step)
    {
        Color originalColor = Color.gray; // Assuming the original color is Unity's standard gray
        Renderer footRenderer;

        // If left step
        if (step == 1) {
            footRenderer = LeftFoot.GetComponent<Renderer>();
            footRenderer.material.color = Color.red;
        } else if (step == 2) {
            footRenderer = RightFoot.GetComponent<Renderer>();
            footRenderer.material.color = Color.red;
        } else {
            yield break; // Exit if step is not 1 or 2
        }

        // Wait for 0.3 seconds
        yield return new WaitForSeconds(0.2f);

        // Revert the color back to original
        footRenderer.material.color = originalColor;
    }

    public void PlayLastRecorded()
    {
        if (isPlaying)
        {
            Debug.Log("Already playing a CSV file. Stopping playback.");
            StopCSV();
            return;
        }
        string[] csvFiles = Directory.GetFiles("Assets/MLTrainingData", "*.csv");
        string latestFile = string.Empty;
        DateTime latestCreationTime = DateTime.MinValue;

        foreach (string file in csvFiles)
        {
            DateTime creationTime = File.GetCreationTime(file);
            if (creationTime > latestCreationTime)
            {
                latestCreationTime = creationTime;
                latestFile = file;
            }
        }

        if (!string.IsNullOrEmpty(latestFile))
        {
            PlayCSV(latestFile);
        }
        else
        {
            Debug.Log("No CSV files found in Assets/MLTrainingData folder.");
        }
    }
    
    IEnumerator ReplayData()
    {
    
        foreach (ReplayFrame frame in replay.frames)
        {
            // Visualize step
            StartCoroutine(VisualizeStep(frame.step));
            // Apply position
            LeftHand.transform.localPosition = frame.leftHandPos;
            RightHand.transform.localPosition = frame.rightHandPos;
            Head.transform.localPosition = frame.headPos;

            // Apply rotation
            LeftHand.transform.localRotation = frame.leftHandRot;
            RightHand.transform.localRotation = frame.rightHandRot;
            Head.transform.localRotation = frame.headRot;
    
            yield return new WaitForSeconds(1f / 20f); // wait for 1/20 second
        }
        isPlaying = false;
    }
}

public class Replay
{
    public List<ReplayFrame> frames = new List<ReplayFrame>();

    public Replay(string filePath)
    {
        using (var reader = new StreamReader(filePath))
        {
            // Skip the first line
            reader.ReadLine();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');
                ReplayFrame frame = new ReplayFrame(values);
                frames.Add(frame);
            }
        }
    }
}

public class ReplayFrame
{
    public int step;
    public Vector3 leftHandPos;
    public Quaternion leftHandRot;
    public Vector3 rightHandPos;
    public Quaternion rightHandRot;
    public Vector3 headPos;
    public Quaternion headRot;

    public ReplayFrame(string[] values)
    {
        if (values.Length < 38)
        {
            return;
        }

        step = int.Parse(values[1]);
        // Updated indices to match the new structure starting at index 20
        leftHandPos = ParseVector3(values, 20);
        leftHandRot = ParseQuaternion(values, 23); // W value set to 1f assuming unit quaternion
        headPos = ParseVector3(values, 26);
        headRot = ParseQuaternion(values, 29); // W value set to 1f assuming unit quaternion
        rightHandPos = ParseVector3(values, 32);
        rightHandRot = ParseQuaternion(values, 35); // W value set to 1f assuming unit quaternion
    }

    private Vector3 ParseVector3(string[] values, int startIndex)
    {
        return new Vector3(
            float.Parse(values[startIndex], CultureInfo.InvariantCulture),
            float.Parse(values[startIndex + 1], CultureInfo.InvariantCulture),
            float.Parse(values[startIndex + 2], CultureInfo.InvariantCulture));
    }

    private Quaternion ParseQuaternion(string[] values, int startIndex)
    {
        return new Quaternion(
            float.Parse(values[startIndex], CultureInfo.InvariantCulture),
            float.Parse(values[startIndex + 1], CultureInfo.InvariantCulture),
            float.Parse(values[startIndex + 2], CultureInfo.InvariantCulture),
            1f); // Assuming unit quaternion with W value set to 1f
    }
}
