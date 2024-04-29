#include "robot_control.h"

RobotControl::RobotControl() {
  // Constructor
}

void RobotControl::init() {
  // Initialize GPIO pins, sensors, and modules
  // Initialize GPIO pins
  pinMode(5, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(2, OUTPUT);
  pinMode(sensorPin, INPUT);
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
  RampInfeed(88);
  }
  FlagWave(0);
  MoveFunnelServo(125);
  // Initialize sensors and modules
  //gyro_init();
  //lineDetect.init();  
}


void RobotControl::MoveFunnelServo(int pos){
	FunnelServo.write(pos);
}
	
void RobotControl::FlagWave(int pos){
	FlagServo.write(pos);
	}
	
int RobotControl::ColorSensor() {
	return analogRead(sensorPin);
}

int RobotControl::GetDist(){
	return TMPFront.getDist();
}


int RobotControl::GetPixyAngle() {
  return lineDetect.getAng(MAX_ANGLE);
}

void RobotControl::pickup() {
	manipulatorControl.pickup();
}

void RobotControl::cruisin() {
	manipulatorControl.cruisin();
}


void RobotControl::RampInfeed(int DesiredAngle){
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

void RobotControl::lineFollow(int robotSpeed, double targetOffset) {

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
	int speedOffset = 2*(100 * (angle) / 90 - lineOffset);  // double to int

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

bool RobotControl::stopTheMotors() {
  // Function to stop the motors
  motorDriver.setSpeed(0, 0);
  for (int i = 0; i < MotorRunDuration; i++) {
    motorDriver.startMove();
  }
  return true;
}

// TODO: Implement crossGap
void RobotControl::crossGap() {

}
double RobotControl::USDistance(){
  float distance_cm;
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // measure duration of pulse from ECHO pin
  distance_cm = pulseIn(ECHO_PIN, HIGH)*.017;


  return distance_cm;
}

void RobotControl::theGobbler(int speed) {
  manipulatorControl.setStorageAngle(125, 0.3);
  manipulatorControl.feedTheBeast(speed);
}


void RobotControl::servo_write(float angle) {
  myservo.write(angle);
}

bool RobotControl::doTheJig() {
  // Function to stop the motors
  motorDriver.setSpeed(-250, 250);
  for (int i = 0; i < 125; i++) {
    motorDriver.startMove();
  }
  motorDriver.setSpeed(250, -250);
  for (int i = 0; i < 125; i++) {
    motorDriver.startMove();
  }
  motorDriver.setSpeed(-250, 250);
  for (int i = 0; i < 125; i++) {
    motorDriver.startMove();
  }
   motorDriver.setSpeed(250, -250);
  for (int i = 0; i < 125; i++) {
    motorDriver.startMove();
  }
  motorDriver.setSpeed(0, 0);
  for (int i = 0; i < MotorRunDuration; i++) {
    motorDriver.startMove();
  }
  
  return true;
}

void RobotControl::constrainTurnSpeed(float &speed) {
  // Constrain the turn speed to a maximum value
  const float MaxTurnSpeed = 200;
  if (speed > MaxTurnSpeed) speed = MaxTurnSpeed;
  if (speed < -MaxTurnSpeed) speed = -MaxTurnSpeed;
}


void RobotControl::turnRight(){
    TickType_t startTick = xTaskGetTickCount();
    TickType_t delayTicks = pdMS_TO_TICKS(250);
    motorDriver.setSpeed(150,-150);
    while ((xTaskGetTickCount() - startTick) < delayTicks) {
        motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
	stopTheMotors();

}

void RobotControl::turnLeft(){
    TickType_t startTick = xTaskGetTickCount();
    TickType_t delayTicks = pdMS_TO_TICKS(250);
    motorDriver.setSpeed(-150,150);
    while ((xTaskGetTickCount() - startTick) < delayTicks) {
        motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
	stopTheMotors();
}


void RobotControl::TurnLeft90() {

  motorDriver.setSpeed(-150, 150);
  for (int i = 0; i < 1400; i++) {
    motorDriver.startMove();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
  motorDriver.setSpeed(0, 0);
  for (int i = 0; i < 500; i++) {
    motorDriver.startMove();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

void RobotControl::TurnLeft45() {

  motorDriver.setSpeed(-150, 150);
  for (int i = 0; i < 700; i++) {
    motorDriver.startMove();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
  motorDriver.setSpeed(0, 0);
  for (int i = 0; i < 500; i++) {
    motorDriver.startMove();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

void RobotControl::TurnRight45() {

  motorDriver.setSpeed(150, -150);
  for (int i = 0; i < 700; i++) {
    motorDriver.startMove();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
  motorDriver.setSpeed(0, 0);
  for (int i = 0; i < 500; i++) {
    motorDriver.startMove();
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

void RobotControl::moveToDetection(Detection detection){
    TickType_t startTick = xTaskGetTickCount();
    TickType_t delayTicks = pdMS_TO_TICKS(150);
    motorDriver.setSpeed(-150,150);
    while ((xTaskGetTickCount() - startTick) < delayTicks) {
        motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
}



void RobotControl::moveForward(){
    TickType_t startTick = xTaskGetTickCount();
    TickType_t delayTicks = pdMS_TO_TICKS(250);
    motorDriver.setSpeed(150,150);
    while ((xTaskGetTickCount() - startTick) < delayTicks) {
        motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
	stopTheMotors();
}



bool RobotControl::turnTo(Detection detection){

	if (detection.horizontal_angle >= 2 || detection.horizontal_angle <= -2) {
		if (detection.horizontal_angle < 0 ) {
			turnLeft();
		} else {
			turnRight();
		}
		return false;
	}
	return true;
}

bool RobotControl::moveTo(Detection detection){
    motorDriver.setSpeed(150,150);
	if (detection.depth_mm >= 210) {
		moveForward();
		return false;
	}
	return true;
    
}


float RobotControl::calculateDistanceToMove(float dist_mm){
    TickType_t startTick = xTaskGetTickCount();
    TickType_t delayTicks = pdMS_TO_TICKS(250);
    motorDriver.setSpeed(-150,150);
    while ((xTaskGetTickCount() - startTick) < delayTicks) {
        motorDriver.startMove();
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
}
