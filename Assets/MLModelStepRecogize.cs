using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using Unity.Sentis.Layers;
using UnityEngine;

public class MLModelStepRecogize : MonoBehaviour
{
    [SerializeField] ModelAsset modelAsset;

    [SerializeField] float[] results;

    private Model runtimeModel;
    private IWorker worker;
    private TensorFloat inputTensor;

    private MotionDebugging motionDebugging;


    void Start()
    {
        //motionDebugging = GetComponent<MotionDebugging>();
        //runtimeModel = ModelLoader.Load(modelAsset);
        //worker = runtimeModel.CreateWorker(runtimeModel);
        runtimeModel = ModelLoader.Load(modelAsset);

        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);

        List<Model.Input> inputs = runtimeModel.inputs;

        // Loop through each input
        foreach (var input in inputs)
        {
            // Log the name of the input, for example Input3
            Debug.Log(input.name);

            // Log the tensor shape of the input, for example (1, 1, 28, 28)
            Debug.Log(input.shape);
        }

    }
    

    


    public float[] ExecuteModel(float[][] input)
    {
        inputTensor?.Dispose();
        // Flatten the 2D array into a 1D array using LINQ
        float[] data = input.SelectMany(subArray => subArray).ToArray();

        // Create a 3D tensor shape with size 1 × 7 × 24
        TensorShape shape = new TensorShape(1, 7, 24);

        // Create a new tensor from the array
        inputTensor = new TensorFloat(shape, data);
        
        // Run the model with the tensor as input
        worker.Execute(inputTensor);

        // Get the output tensor
        TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;
        outputTensor.CompleteOperationsAndDownload();

        // Get the output data as a read-only array
        float[] outputData = outputTensor.ToReadOnlyArray();
        return outputData;
    }

    void Update()
    {
        
    }


}
