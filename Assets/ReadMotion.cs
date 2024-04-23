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
    void Start()
    {
        replay = new Replay(@"C:\Users\zidm\Repo\VR-Motion-Interpreter\2024011705-0110.csv");
        StartCoroutine(ReplayData());
    }

    // Update is called once per frame
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
