#ifndef LineFollower_h
#define LineFollower_h

#include <Arduino.h>
#include <MotorDriver.h>
#include <pixy_line_detection.h>
//#include <gyro.h>

class LineFollower {
public:
  LineFollower();
  void begin();
  bool followLineUntilTurn();
  void turn(float turnAngle);

private:
  MotorDriver motorDriver;
  PixyLineDetect lineDetect;
  const int MaxAngle = 90;
  const int BaseSpeed = 125;
  const int MotorRunDuration = 500;

  void stopMotors();
  void constrainTurnSpeed(float &speed);
};

#endif
