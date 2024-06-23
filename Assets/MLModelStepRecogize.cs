using System.Collections;
using System.Collections.Generic;
using System.Linq;
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
        //motionDebugging = GetComponent<MotionDebugging>();
        //runtimeModel = ModelLoader.Load(modelAsset);
        //worker = runtimeModel.CreateWorker(runtimeModel);
        Model runtimeModel = ModelLoader.Load(modelAsset);

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
    

    


    public int ExecuteModel(float[][] input)
    {
        // Flatten the 2D array into a 1D array using LINQ
        float[] data = input.SelectMany(subArray => subArray).ToArray();

        // Create a 3D tensor shape with size 1 × 7 × 24
        TensorShape shape = new TensorShape(1, 7, 24);

        // Create a new tensor from the array
        TensorFloat tensor = new TensorFloat(shape, data);

        // Create an inference engine (a worker)
        IWorker worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
        
        // Run the model with the tensor as input
        worker.Execute(inputTensor);

        // Get the output tensor
        TensorInt outputTensor = worker.PeekOutput() as TensorInt;

        // Print the output tensor
        Debug.Log(outputTensor);

        return outputTensor == null ? -1 : outputTensor[0];
    }

    void Update()
    {
        
    }


}
