using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering.Universal;
using UnityEngine.UI;


public class StepFeedback : MonoBehaviour
{
    [SerializeField] Image feedbackScreen;

    [SerializeField] public bool rightStepped;
    [SerializeField] public bool leftStepped;
    bool _rightStepped = false;
    bool _leftStepped = false;
    [SerializeField] float stepTimer = 0.3f;
    float _rightStepTimer = 0.0f;
    float _leftStepTimer = 0.0f;

    void Update()
    {
        bool justStepped = (rightStepped || leftStepped);
        bool stepped = (justStepped || _rightStepped || _leftStepped);

        if (rightStepped){
            rightStepped = false;
            _rightStepped = true;
            feedbackScreen.color = Color.red;

            _rightStepTimer = stepTimer;
        }
        if (leftStepped){
            leftStepped = false;
            _leftStepped = true;
            feedbackScreen.color = Color.blue;

            _leftStepTimer = stepTimer;
        }
        if (_rightStepped){
            _rightStepTimer -= Time.deltaTime;
            if (_rightStepTimer <= 0){
                _rightStepped = false;
            }
        }
        if (_leftStepped){
            _leftStepTimer -= Time.deltaTime;
            if (_leftStepTimer <= 0){
                _leftStepped = false;
            }
        }
        if (!stepped){
            feedbackScreen.color = Color.gray;
        }
    }
}
