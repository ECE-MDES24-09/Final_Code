
#ifndef ROBOT_CONTROL_H
#define ROBOT_CONTROL_H

// Including necessary libraries and other header files
#include <Arduino.h>              // Standard Arduino library
#include <MotorDriver.h>          // Custom library for motor control
#include <pixy_line_detection.h>  // Custom library for line detection using Pixy camera
#include "Manipulator_Control.h"
#include <DetectionsBuffer.h>
//#include <gyro.h>                 // Custom library for gyroscopic sensor
#include <TMP.h>
#define MAX_ANGLE 70.0
#define SPEED_MULTIPLIER 1.0
#define ROLLER_MULTIPLIER 1.0
#define TRIG_PIN 26
#define ECHO_PIN 27
#define STORAGE_SERVO_PIN 27

// Definition of the RobotControl class
class RobotControl {
public:
  MotorDriver motorDriver;        // Motor driver to control robot's movement
  PixyLineDetect lineDetect;      // Line detection object using Pixy camera
  Manipulator_Control manipulatorControl;
  double USDistance();
  TMP TMPFront;
  // Constructor declaration
  RobotControl();
  int CurrentInfeedAngle;
  void theGobbler(int speed);
  // Initializes the RobotControl
  void init();
  int GetPixyAngle();
  int ColorSensor();
  // Follows a line until a turn is detected, returns true if turn is detected
  bool followLineUntilTurn();
  
  // Stops all motors
  bool stopTheMotors();
  
  // Starts all motors
  bool startTheMotors();
  
  bool doTheJig();
  
  bool takeItBack();
  
  bool circleLeft();
  
  bool circleRight();
	
  // Makes a turn with a specified angle
  void turn(float turnAngle);

  void lineFollow(int robotSpeed, double targetOffset);
  void crossGap();
  void pickup();
  void cruisin();
  void turnRight();
  void turnLeft();
  void findLine();
  void TurnLeft90();
  void TurnLeft45();
  void TurnRight45();
  void moveToDetection(Detection detection);
  void moveForward();
  bool turnTo(Detection detection);
  bool moveTo(Detection detection);
  float calculateDistanceToMove(float dist_mm);
  void RampInfeed(int DesiredAngle);
  void MoveFunnelServo(int);
  void FlagWave(int pos);
  int GetDist();
  Servo myservo;
  Servo FlagServo;
  Servo FunnelServo;
  void servo_write(float angle);

private:
  // Objects for motor control and sensors

  //TMP TMPFront;
  const int MaxAngle = 90;        // Maximum angle for turning
  const int BaseSpeed = 125;      // Base speed for the motors
  const int MotorRunDuration = 500; // Duration for running the motors in milliseconds
  const int sensorPin = 12;
  // Stops all motors
  void stopMotors();

  // Constrains the turn speed to a safe range
  void constrainTurnSpeed(float &speed);
};

// End of include guard
#endif
