#include "Manipulator_Control.h"

// Initialize the manipulator control (attach servos)
void Manipulator_Control::init() {
    storageServo.attach(STORAGE_SERVO_PIN);
    pinMode(INFEED_LEFT_PIN, OUTPUT);
    pinMode(OUTFEED_LEFT_PIN, OUTPUT);
    pinMode(INFEED_RIGHT_PIN, OUTPUT);
    pinMode(OUTFEED_RIGHT_PIN, OUTPUT);

}

// Set the angle for the storage servo with speed control
void Manipulator_Control::setStorageAngle(float angle, float speedFactor) {
    storageServo.write(angle);
}

// Set speed for the left roller servo
void Manipulator_Control::setLeftRollerSpeed(int speed) {
    leftRollerServo.writeMicroseconds(map(speed, -100, 100, 1000, 2000)); // Assuming speed range is -100 to 100
}

// Set speed for the right roller servo
void Manipulator_Control::setRightRollerSpeed(int speed) {
    rightRollerServo.writeMicroseconds(map(speed, -100, 100, 2000, 1000)); // Assuming speed range is -100 to 100
}

void Manipulator_Control::feedTheBeast(int speed) {
	if(speed > 0){
			analogWrite(INFEED_LEFT_PIN, speed);
			analogWrite(INFEED_RIGHT_PIN, speed);

	}
	if(speed < 0){
			analogWrite(OUTFEED_LEFT_PIN, speed);
			analogWrite(OUTFEED_RIGHT_PIN, speed);

	}
}

void Manipulator_Control::cruisin() {
	setStorageAngle(crusingAng, 0.3);
}

void Manipulator_Control::pickup() {
	setStorageAngle(pickUpAng, 0.3);
}

// Update the angle of the storage servo gradually
void Manipulator_Control::updateStorageAngle(float speedFactor) {
    // Check if the current angle is not equal to the target angle
    if (currentStorageAngle != targetStorageAngle) {
        // If the current angle is less than the target angle
        if (currentStorageAngle < targetStorageAngle) {
            currentStorageAngle += speedFactor; // Increment the current angle

            // Check if the current angle has surpassed the target angle
            if (currentStorageAngle > targetStorageAngle) {
                currentStorageAngle = targetStorageAngle; // Limit the current angle to the target angle
            }
        } 
        // If the current angle is greater than the target angle
        else {
            currentStorageAngle -= speedFactor; // Decrement the current angle

            // Check if the current angle has surpassed the target angle
            if (currentStorageAngle < targetStorageAngle) {
                currentStorageAngle = targetStorageAngle; // Limit the current angle to the target angle
            }
        }
        storageServo.write(currentStorageAngle); // Write the updated angle to the storage servo
    }
}
