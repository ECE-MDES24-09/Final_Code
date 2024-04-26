#ifndef MOTOR_DRIVER_H
#define MOTOR_DRIVER_H

// motor pins
#define RIGHT_FORWARD_PIN 3
#define RIGHT_BACKWARD_PIN 2
#define LEFT_FORWARD_PIN 5
#define LEFT_BACKWARD_PIN 4

#define INFEED_LEFT_IN_PIN 8
#define INFEED_LEFT_OUT_PIN 9
#define INFEED_RIGHT_IN_PIN 10
#define INFEED_RIGHT_OUT_PIN 11

#define RAMP_RATE 1
#define RAMPING true

#include "Arduino.h"


class MotorDriver {

public:

	void setSpeed(int targetLeftSpeed, int targetRightSpeed) {
		this->targetLeftSpeed = targetLeftSpeed;
		this->targetRightSpeed = targetRightSpeed;
	}


	
//No Ramping
	void Infeed(int speed){
		
		if(speed > 0){
				analogWrite(INFEED_LEFT_IN_PIN, speed);
				analogWrite(INFEED_RIGHT_IN_PIN, speed);
				analogWrite(INFEED_LEFT_OUT_PIN, 0);
				analogWrite(INFEED_RIGHT_OUT_PIN, 0);

		}
		else if(speed < 0){
				analogWrite(INFEED_LEFT_OUT_PIN, speed);
				analogWrite(INFEED_RIGHT_OUT_PIN, speed);
				analogWrite(INFEED_LEFT_IN_PIN, 0);
				analogWrite(INFEED_RIGHT_IN_PIN, 0);
		}
		else{
				analogWrite(INFEED_LEFT_OUT_PIN, 0);
				analogWrite(INFEED_RIGHT_OUT_PIN, 0);
				analogWrite(INFEED_LEFT_IN_PIN, 0);
				analogWrite(INFEED_RIGHT_IN_PIN, 0);
		}
	}
	
	void startMove() {

		if (RAMPING) {
			// ramping
			if (targetRightSpeed > this->rightSpeed)
				this->rightSpeed += RAMP_RATE;
			else if (targetRightSpeed < this->rightSpeed)
				this->rightSpeed -= RAMP_RATE;

			if (targetLeftSpeed > this->leftSpeed)
				this->leftSpeed += RAMP_RATE;
			else if (targetLeftSpeed < this->leftSpeed)
				this->leftSpeed -= RAMP_RATE;
		}
		else {
			this->leftSpeed = targetLeftSpeed;
			this->rightSpeed = targetRightSpeed;
		}

		// output PWMs
		if (rightSpeed >= 0) {
			analogWrite(RIGHT_FORWARD_PIN, rightSpeed);
			analogWrite(RIGHT_BACKWARD_PIN, 0);
		}
		else {
			analogWrite(RIGHT_FORWARD_PIN, 0);
			analogWrite(RIGHT_BACKWARD_PIN, -rightSpeed);
		}

		if (leftSpeed >= 0) {
			analogWrite(LEFT_FORWARD_PIN, leftSpeed);
			analogWrite(LEFT_BACKWARD_PIN, 0);
		}
		else {
			analogWrite(LEFT_FORWARD_PIN, 0);
			analogWrite(LEFT_BACKWARD_PIN, -leftSpeed);
		}




	}

private:

	int leftSpeed = 0;
	int rightSpeed = 0;

	int targetLeftSpeed = 0;
	int targetRightSpeed = 0;

};

#endif