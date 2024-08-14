using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine;

public class MLModelStepRecogize : MonoBehaviour
{
    [SerializeField] ModelAsset modelAsset;

    [SerializeField] float[] results;

    private Model runtimeModel;
    private IWorker worker;
    private TensorFloat inputTensor;

    void Awake()
    {
        // If the model asset is null, look for a model asset in "Assets\MLModels" and pick the latest one
        if (modelAsset == null)
        {
            // Get the latest .onnx file in the "Assets/MLModels" folder by creation time
            string latestModelPath = System.IO.Directory.GetFiles("Assets/MLModels", "*.onnx")
                .OrderByDescending(f => new System.IO.FileInfo(f).CreationTime)
                .FirstOrDefault();
            
            if (!string.IsNullOrEmpty(latestModelPath))
            {
                string absoluteModelPath = System.IO.Path.GetFullPath(latestModelPath);
                Debug.Log("No model asset specified, using the latest model: " + absoluteModelPath);
                
                // Convert the path to a model asset
                runtimeModel = ModelLoader.Load(absoluteModelPath);
            }
            else
            {
                Debug.Log("No .onnx model files found in the Assets/MLModels directory.");
            }

        } else
        {
            // Load the model asset
            runtimeModel = ModelLoader.Load(modelAsset);
        }

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
