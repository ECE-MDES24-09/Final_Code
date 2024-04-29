#ifndef MOTOR_DRIVER_H
#define MOTOR_DRIVER_H

// motor pins
#define RIGHT_FORWARD_PIN 5
#define RIGHT_BACKWARD_PIN 4
#define LEFT_FORWARD_PIN 3
#define LEFT_BACKWARD_PIN 2

#define RAMP_RATE 2 
#include <Arduino_FreeRTOS.h>
#include "Arduino.h"

class MotorDriver {

public:
	void init();
	void setSpeed(int targetLeftSpeed, int targetRightSpeed);
	void startMove();
	void FollowLine(int right, int left, double angle, double offset);
	int OS = 0;

	int leftSpeed = 0;
	int rightSpeed = 0;

	int targetLeftSpeed = 0;
	int targetRightSpeed = 0;

private:



};

#endif

