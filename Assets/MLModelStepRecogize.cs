using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;

public class MLModelStepRecogize : MonoBehaviour
{
    [SerializeField] ModelAsset modelAsset;

    [SerializeField] GameObject LeftHand;
    [SerializeField] GameObject RightHand;
    [SerializeField] GameObject Head;

    [SerializeField] float[] results;

    private Model runtimeModel;
    private IWorker worker;
    private TensorFloat inputTensor;

    private MotionDebugging motionDebugging;

    void Start()
    {
        motionDebugging = GetComponent<MotionDebugging>();
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = runtimeModel.CreateWorker(runtimeModel);
    }

    


    private void ExecuteModel(float[][] input)
    {
        // Determine the dimensions of the tensor
        int batchSize = input.Length;
        int featureSize = input[0].Length;
        // Create a single-dimensional array to hold the data
        float[] singleDimensionData = new float[batchSize * featureSize];
        // Copy the data from the 2D array to the 1D array
        for (int i = 0; i < batchSize; i++)
        {
            System.Array.Copy(input[i], 0, singleDimensionData, i * featureSize, featureSize);
        }

        // Create the tensor from the 1D array
        inputTensor = new TensorFloat(new TensorShape(batchSize, featureSize), singleDimensionData);

    }
    void Update()
    {
        
    }


}
