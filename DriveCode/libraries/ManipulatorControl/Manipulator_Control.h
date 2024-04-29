#ifndef MANIPULATOR_CONTROL_H
#define MANIPULATOR_CONTROL_H

#include "Arduino.h"
#include <Servo.h>
#include <Arduino_FreeRTOS.h>
#define INFEED_LEFT_PIN 8
#define OUTFEED_LEFT_PIN 9
#define INFEED_RIGHT_PIN 10
#define OUTFEED_RIGHT_PIN 11



class Manipulator_Control {
public:
    void init();
	void pickup();
	void cruisin();
    void setStorageAngle(float angle, float speedFactor); // Function to set storage angle with speed control
    void setLeftRollerSpeed(int speed); // Function to set left roller speed
    void setRightRollerSpeed(int speed); // Function to set right roller speed
	void feedManipulator(int speed);
    const int STORAGE_SERVO_PIN = 12;
	void feedTheBeast(int speed);

private:

    Servo storageServo;
    Servo leftRollerServo;
    Servo rightRollerServo;

    float currentStorageAngle; // Current angle of the storage servo
    float targetStorageAngle; // Target angle for the storage servo

    const float CubeDropAng = 115.0f;
    const float pickUpAng = 120.0f;
    const float crusingAng = 90.0f;
    const float fuelDropAng = 75.0f;

    // Function to update the storage servo angle gradually
    void updateStorageAngle(float speedFactor);
};

#endif
