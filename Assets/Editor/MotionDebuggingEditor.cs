using UnityEngine;
using UnityEditor;
using UnityEngine.InputSystem;

[CustomEditor(typeof(MotionDebugging))]
public class MotionDebuggingEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        MotionDebugging myScript = (MotionDebugging)target;
        if (GUILayout.Button("Toggle Recording"))
        {
            myScript.toggleRecording(new InputAction.CallbackContext());  
        }
        if (GUILayout.Button("Toggle Recognition"))
        {
            myScript.toggleRecognition(new InputAction.CallbackContext());  
        }
    }
}