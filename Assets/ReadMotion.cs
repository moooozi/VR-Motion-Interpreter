using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Globalization;

public class ReadMotion : MonoBehaviour
{
    [SerializeField] GameObject LeftHand;
    [SerializeField] GameObject RightHand;
    [SerializeField] GameObject Head;
    private Replay replay;

    // Start is called before the first frame update
    public void PlayCSV(string file_path)
    {
        replay = new Replay(file_path);
        StartCoroutine(ReplayData());
    }

    public void StopCSV()
    {
        StopAllCoroutines();
    }

    public void PlayLastRecorded()
    {
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
            LeftHand.transform.position = frame.leftHandPos;
            LeftHand.transform.rotation = frame.leftHandRot;
            RightHand.transform.position = frame.rightHandPos;
            RightHand.transform.rotation = frame.rightHandRot;
            Head.transform.position = frame.headPos;
            Head.transform.rotation = frame.headRot;
            yield return new WaitForSeconds(1f / 20f); // wait for 1/20 second
        }
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
    public Vector3 leftHandPos;
    public Quaternion leftHandRot;
    public Vector3 rightHandPos;
    public Quaternion rightHandRot;
    public Vector3 headPos;
    public Quaternion headRot;

    public ReplayFrame(string[] values)
    {
        // Updated indices to match the new structure starting at index 21
        leftHandPos = new Vector3(float.Parse(values[20], CultureInfo.InvariantCulture), float.Parse(values[21], CultureInfo.InvariantCulture), float.Parse(values[22], CultureInfo.InvariantCulture));
        leftHandRot = new Quaternion(float.Parse(values[23], CultureInfo.InvariantCulture), float.Parse(values[24], CultureInfo.InvariantCulture), float.Parse(values[25], CultureInfo.InvariantCulture), 1f); // W value set to 1f assuming unit quaternion
        headPos = new Vector3(float.Parse(values[26], CultureInfo.InvariantCulture), float.Parse(values[27], CultureInfo.InvariantCulture), float.Parse(values[28], CultureInfo.InvariantCulture));
        headRot = new Quaternion(float.Parse(values[29], CultureInfo.InvariantCulture), float.Parse(values[30], CultureInfo.InvariantCulture), float.Parse(values[31], CultureInfo.InvariantCulture), 1f); // W value set to 1f assuming unit quaternion
        rightHandPos = new Vector3(float.Parse(values[32], CultureInfo.InvariantCulture), float.Parse(values[33], CultureInfo.InvariantCulture), float.Parse(values[34], CultureInfo.InvariantCulture));
        rightHandRot = new Quaternion(float.Parse(values[35], CultureInfo.InvariantCulture), float.Parse(values[36], CultureInfo.InvariantCulture), float.Parse(values[37], CultureInfo.InvariantCulture), 1f); // W value set to 1f assuming unit quaternion
    }
}
