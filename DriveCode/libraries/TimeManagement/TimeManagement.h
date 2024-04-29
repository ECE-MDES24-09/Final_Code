#ifndef TimeManagement_h
#define TimeManagement_h

#include <Arduino.h>
#include <Arduino_FreeRTOS.h>
#include <timers.h>
#include <semphr.h>
#include <task.h>
#include <DetectionsBuffer.h>



class TimeManagement {
public:
	
  bool timeOut = false;
  TimerHandle_t StateTimeoutTimer;
  TaskHandle_t MotorBoxTaskHandle;
	
   TimeManagement::TimeManagement(); // Constructor
  ~TimeManagement(); // Destructor
  void TimeManagement::startState(int stateId);
  void TimeManagement::endState(int stateId, bool allocate);
  void TimeManagement::allocateTime(int stateId);
  long TimeManagement::getRemainingTime() const; 
  void TimeManagement::setTimeForState(int state, long time); 
  long TimeManagement::getRemainingTimeForState(int stateId) const; 
  bool TimeManagement::isMaxTimeSufficient() const;
  void TimeManagement::startTimer();
  long TimeManagement::getTimeBank() const;
  long TimeManagement::getStateLimit(int stateId) const;
  static void StateTimeoutCallback(TimerHandle_t xTimer);
  long TimeManagement::getRunTime() const;


private:
  // WAIT_FOR_START ID == 0. It does not count toward timer.
  long _startTime;
  long _maxTime = 105000;
  // With Buffer State Limits
  //long _stateTimeLimits[19] = { 0, 10000, 10000, 5000, 4000, 1500, 2000, 1500, 5000, 3000, 10000, 5000, 15000, 5000, 15000, 2000, 3000, 0, 0 }; 
  // Without Buffer State Limits
  long _stateTimeLimits[19] = { 0, 10000, 11000, 7000, 4000, 3000, 2000, 3000, 5000, 3000, 10000, 5000, 16000, 5000, 16000, 2000, 3000, 0, 0 }; 
  // Without Buffer State Limits - Timout Test
  //long _stateTimeLimits[19] = { 0, 1000, 1100, 700, 400, 300, 200, 300, 500, 300, 1000, 500, 1600, 500, 1600, 200, 3000, 0, 0 }; 
  int _statePriorities[19] = {0, 6, 7, 5, 3, 2, 1, 2, 4, 2, 6, 4, 8, 4, 8, 1, 2, 0, 0};
  int _numStates = 19;
  int _currentState; 
  long _stateStartTime;
  long _stateEndTime;
  long _currentstateMaxTime;
  long _timeBank;
  const long STATE_SWITCH_BUFFER = 500; // 0.5 seconds in milliseconds
  bool _isStartStateCompleted;
  const int WAIT_FOR_START_STATE_ID = 0;
  TimerHandle_t* _stateTimers;
  TimerHandle_t _bufferTimer;
  
  
  // Timer handle
  
};

#endif