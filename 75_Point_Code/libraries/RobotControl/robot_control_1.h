#ifndef ROBOT_CONTROL_H
#define ROBOT_CONTROL_H

#define MAX_ANGLE 70.0
#define SPEED_MULTIPLIER 1.0
#define ROLLER_MULTIPLIER 1.0
#define STORAGE_SERVO_PIN 12
#define TRIG_PIN 27
#define ECHO_PIN 26

#include "Arduino.h"
#include <Servo.h>
#include <motor_driver.h>
#include <pixy_line_detection1.h>
#include <serial_communication.h>

//#include <gyro_blue.h>
#include <TMP.h>
#include <Servo.h>

// Counter and compare values
const uint16_t t1_load = 0;
const uint16_t t1_comp = 250;

enum class RollerState
{
	InTake,
	OutTake,
	Stop,
};

enum class StorageState
{
	Start,
	PickUp,
	BigCubeDrop,
	SmallCubeDrop,
	FuelDrop,
	Cruise,
};

class RobotControl {
	

public:
	PixyLineDetect lineDetect;
	MotorDriver motorDriver;
	TMP TMPFront;
	//Gyro gyro;

	int CurrentInfeedAngle;
	const int sensorPin = A12; 
	
	Servo myservo;
	Servo FlagServo;
	Servo FunnelServo;
	
	void init() {
		
		FlagServo.attach(49);
		FunnelServo.attach(53);
		CurrentInfeedAngle = 125;
		
		TMPFront.init(SerialSelect::S0);
		lineDetect.init();
		myservo.attach(STORAGE_SERVO_PIN);
		//gyro.init();

		pinMode(TRIG_PIN, OUTPUT);
		pinMode(ECHO_PIN, INPUT);
		for(int i = 0; i<40;i++){
		RampInfeed(115);
		}
		FlagWave(0);
		MoveFunnelServo(125);
		lineDetect.lineMode();
		
	}
	
	void MoveFunnelServo(int pos){
	FunnelServo.write(pos);
	}
	
	void FlagWave(int pos){
	FlagServo.write(pos);
	}
	
	int ColorSensor() {
		return analogRead(sensorPin);
	}
	int GetDist(){
		return TMPFront.getDist();
	}
	int GetPixyAngle(){
		lineDetect.update();
		return lineDetect.getAng(MAX_ANGLE);
	}
	
	double USDistance(){
	float distance_cm;
	digitalWrite(TRIG_PIN, HIGH);
	delayMicroseconds(10);
	digitalWrite(TRIG_PIN, LOW);

	// measure duration of pulse from ECHO pin
	distance_cm = pulseIn(ECHO_PIN, HIGH)*.017;


	return distance_cm;
	}

	void test(){
		
		//for(int i = 0; i<37;i++){
		//RampInfeed(88);
		//delay(15);
		//}
		//delay(500);
		for(int i = 0; i<37;i++){
		RampInfeed(125);

		}
		//delay(1500);
		//motorDriver.Infeed(255);
		//delay(1500);
		//motorDriver.Infeed(0);
		
		//moves infeed motors
		//motorDriver.Infeed(-250);
		
		//color sensor
		//Serial.println(ColorSensor());
		
		//Ultrasonic
		Serial.println(USDistance());
		
		//lidar
		//Serial.print(TMPFront.getDist());
		
		//moves block picking up thing
		//myservo.write(125); //88 high 130 low
		
		//test line following
		//lineFollow(130, 0);

		//lineDetect.update();
		//Serial.println(lineDetect.getAng(MAX_ANGLE));
		
		//tests drive train
		/*
		motorDriver.setSpeed(-150,150);
		for(int i = 0; i<200; i++){
		motorDriver.startMove();
		delay(1);
		}
		motorDriver.setSpeed(0,0);
		for(int i = 0; i<500; i++){
		motorDriver.startMove();
		delay(1);
		}
		*/
	}
	/*
	void Turn(float Turn){
			
		float Angle = Gyro.getYaw();
		float TurnSpeed = 0, FinalAngle = Angle + Turn;
		if (FinalAngle > 360) FinalAngle = FinalAngle - 360;
		if (FinalAngle < 0) FinalAngle = FinalAngle + 360;
		
		while(abs(Angle - FinalAngle) > 2){

		TurnSpeed = 1.8*(Angle - FinalAngle);
		if(TurnSpeed > 120) TurnSpeed = 120;
        if(TurnSpeed < -120) TurnSpeed = -120;
        if(TurnSpeed > 0 && TurnSpeed < 50) TurnSpeed = 50;
        if(TurnSpeed > -50 && TurnSpeed < 0) TurnSpeed = -50;

        Serial.print(Angle);
        Serial.print(" ");
        Serial.print(FinalAngle);
        Serial.print(" ");
        Serial.println(TurnSpeed);

		motorDriver.setSpeed(TurnSpeed, -TurnSpeed);
        motorDriver.startMove();

		Angle = Gyro.getYaw();	
		if(Angle <0) Angle = Angle + 360;		
		}
		
    	motorDriver.setSpeed(0, 0);
		for(int i = 255;i>0;i--){
			motorDriver.startMove();
		}
    }
	
	*/
	
	void RampInfeed(int DesiredAngle){
		bool SkipDelay = true;
		if (DesiredAngle > CurrentInfeedAngle){
			CurrentInfeedAngle++;
			SkipDelay = false;
		}
		else if (DesiredAngle < CurrentInfeedAngle){
			CurrentInfeedAngle--;
			SkipDelay = false;
		}
		myservo.write(CurrentInfeedAngle);
		if(SkipDelay = false){
		delay(15);
		}
	}

	void lineFollow(int robotSpeed, double targetOffset) {

		lineDetect.update();

		double angle = lineDetect.getAng(MAX_ANGLE);
		double lineOffset = lineDetect.getOffset();
		
		//Serial.println(angle);

		// compute target angle and speed offset
		//++
		//double sensitivity = 1 / 300.0;
		//double ossen = lineOffset * sensitivity;
		//if (ossen > 1.0) ossen = 1.0;
		//else if (ossen < -1.0) ossen = -1.0;
		//double targetAng = asin(ossen) * 180 / M_PI;
		//double angleDiff = angle - targetAng;
		//int speedOffset = static_cast<int>(100.0 * (angleDiff) / 90.0);

		// another way
		int speedOffset = (100 * (angle) / 90 - lineOffset);  // double to int

		int leftSpeed = robotSpeed + speedOffset;
		int rightSpeed = robotSpeed - speedOffset;

		if (leftSpeed > 255)
			leftSpeed = 255;
		else if (leftSpeed < -255)
			leftSpeed = -255;

		if (rightSpeed > 255)
			rightSpeed = 255;
		else if (rightSpeed < -255)
			rightSpeed = -255;

		 Serial.print(leftSpeed);
		 Serial.print("  ");
		 Serial.println(rightSpeed);

		analogWrite(5,leftSpeed);
		analogWrite(3,rightSpeed);


	}

	void manualControl() {

		// Wheels control
		int leftSpeed = dataRecieved[0] * 2 + 1;
		int rightSpeed = dataRecieved[1] * 2 + 1;

		leftSpeed = (int)(leftSpeed * SPEED_MULTIPLIER + 0.5);
		rightSpeed = (int)(rightSpeed * SPEED_MULTIPLIER + 0.5);

		if (leftSpeed <= 20 && leftSpeed >= -20)
			leftSpeed = 0;

		if (rightSpeed <= 20 && rightSpeed >= -20)
			rightSpeed = 0;

		// Serial.print(leftSpeed);
		// Serial.print(rightSpeed);
		motorDriver.setSpeed(leftSpeed, rightSpeed);


		// Roller control
		if (dataRecieved[3] == 1 && lastDataReceived3 == 0)
			HandleSquareButtonPress();

		lastDataReceived3 = dataRecieved[3]; // Update lastDataReceived3 for the next cycle

		if (dataRecieved[4] == 1 && lastDataReceived4 == 0)
			HandleCrossButtonPress();

		lastDataReceived4 = dataRecieved[4]; // Update lastDataReceived3 for the next cycle

		// Storage control
		if (dataRecieved[2] == 1 && lastDataReceived2 == 0)
			HandleTriangleButtonPress();

		lastDataReceived2 = dataRecieved[2]; // Update lastDataReceived2 for the next cycle

	}

	void connectionCheck() {

		// stop the motors if controller is disconnected
		if (millis() - lastTimeRecevieData > 100) {
			for (byte i = 0; i < DATA_LENGTH; i++) {
				dataRecieved[i] = 0;
			}
		}

	}
	
	void updateSerialInput() {

		// write to data received

		int8_t* temp_data = getSerialData(); // Attempt to get data after "START"

		// If data is received and is not just zeros (check for valid data)
		if (temp_data[0] != 0 || temp_data[1] != 0) {
			for (byte i = 0; i < DATA_LENGTH; i++)
				dataRecieved[i] = temp_data[i];
		}

	}

	// InTake
	void HandleSquareButtonPress()
	{
		switch (rollersState)
		{
		case RollerState::Stop:
			rollersState = RollerState::InTake;
			//LeftRollerJoint.targetAngularVelocity = new Vector3(0, RollersSpeed, 0);
			//RightRollerJoint.targetAngularVelocity = new Vector3(0, -RollersSpeed, 0);
			break;
		case RollerState::OutTake:
			rollersState = RollerState::Stop;
			//LeftRollerJoint.targetAngularVelocity = new Vector3(0, 0, 0);
			//RightRollerJoint.targetAngularVelocity = new Vector3(0, 0, 0);
			break;
		}
	}

	// OutTake
	void HandleCrossButtonPress()
	{
		switch (rollersState)
		{
		case RollerState::Stop:
			rollersState = RollerState::OutTake;
			//LeftRollerJoint.targetAngularVelocity = new Vector3(0, -RollersSpeed, 0);
			//RightRollerJoint.targetAngularVelocity = new Vector3(0, RollersSpeed, 0);
			break;
		case RollerState::InTake:
			rollersState = RollerState::Stop;
			//LeftRollerJoint.targetAngularVelocity = new Vector3(0, 0, 0);
			//RightRollerJoint.targetAngularVelocity = new Vector3(0, 0, 0);
			break;
		}
	}

	// Storage tilting control
	void HandleTriangleButtonPress()
	{
		// Based on the current step in the cycle, move to the next state
		switch (storageCycleStep)
		{
		case 0: // Currently in the first Cruise
			storageState = StorageState::PickUp;
			targetStorageAng = pickUpAng; // Set target angle to pick up position
			storageCycleStep++; // Move to the next step in the cycle
			break;
		case 1: // After PickUp, move back to Cruise
			storageState = StorageState::Cruise;
			targetStorageAng = crusingAng; // Set target angle for cruising
			storageCycleStep++; // Move to the next step in the cycle
			break;
		case 2: // In the second Cruise, move to FuelDrop
			storageState = StorageState::FuelDrop;
			targetStorageAng = fuelDropAng; // Set target angle for fuel drop
			storageCycleStep++; // Prepare for the final transition back to Cruise
			break;
		case 3: // After FuelDrop, return to Cruise and reset cycle
			storageState = StorageState::Cruise;
			targetStorageAng = crusingAng; // Set target angle for cruising
			storageCycleStep = 0; // Reset the cycle
			break;
		default:
			// Just in case, reset the cycle if it's out of expected range
			storageCycleStep = 0;
			break;
		}
	}


	void printStates() {
		String message = "Rollers State: ";

		// Append the current state of rollersState to the message
		switch (rollersState) {
		case RollerState::InTake:
			message += "InTake";
			break;
		case RollerState::OutTake:
			message += "OutTake";
			break;
		case RollerState::Stop:
			message += "Stop";
			break;
		default:
			message += "Unknown";
			break;
		}

		message += ", Storage State: ";

		// Append the current state of storageState to the message
		switch (storageState) {
		case StorageState::Start:
			message += "Start";
			break;
		case StorageState::PickUp:
			message += "PickUp";
			break;
		case StorageState::BigCubeDrop:
			message += "BigCubeDrop";
			break;
		case StorageState::SmallCubeDrop:
			message += "SmallCubeDrop";
			break;
		case StorageState::FuelDrop:
			message += "FuelDrop";
			break;
		case StorageState::Cruise:
			message += "Cruise";
			break;
		default:
			message += "Unknown";
			break;
		}

		// Print the entire message on a single line
		Serial.println(message);
	}

	void servo_write() {
		myservo.write(targetStorageAng);
	}

private:



	int8_t dataRecieved[DATA_LENGTH];

	RollerState rollersState = RollerState::Stop;
	StorageState storageState = StorageState::Start;

	float targetStorageAng = 90.0f;
	float storageAng = 0.0f;
	float storageRotateSpeed = 0.1f; // degree per loop
	float bigCubeDropAng = 115.0f;
	float smallCubeDropAng = 115.0f;
	float pickUpAng = 115.0f;
	float crusingAng = 90.0f;
	float fuelDropAng = 75.0f;
	float startAng = 90.0f;

	int8_t lastDataReceived2 = -1; // To track the last state of dataRecieved[2]
	int8_t lastDataReceived3 = -1; // To track the last state of dataRecieved[3]
	int8_t lastDataReceived4 = -1; // To track the last state of dataRecieved[4]
	

	int storageCycleStep = 0;

};


#endif