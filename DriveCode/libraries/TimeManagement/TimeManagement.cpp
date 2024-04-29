#include "TimeManagement.h"

void StateTimerCallback(TimerHandle_t xTimer);
void BufferTimerCallback(TimerHandle_t xTimer);



TimeManagement::TimeManagement(): _isStartStateCompleted(false) {
	bool maxSuff = isMaxTimeSufficient();
	Serial.println(maxSuff);
   _stateTimers = new TimerHandle_t[19];
   _bufferTimer = NULL;
   _timeBank = 0;
   StateTimeoutTimer = xTimerCreate("StateTimeout", pdMS_TO_TICKS(150000), pdFALSE, (void *)this, StateTimeoutCallback);
	
	
  if (StateTimeoutTimer == NULL) {
      Serial.println("Failed to create StateTimeoutTimer");
    } else {
      Serial.println("Created StateTimeoutTimer");
    }


}


TimeManagement::~TimeManagement() {
	for (int i = 0; i < _numStates; ++i) {
		xTimerDelete(_stateTimers[i], portMAX_DELAY);
    }
    delete[] _stateTimers;
    delete[] _stateTimeLimits;
}


bool TimeManagement::isMaxTimeSufficient() const {
    long totalStateTime = 0;
    for (int i = 0; i < _numStates; ++i) {
        totalStateTime += _stateTimeLimits[i];
    }
    return _maxTime >= totalStateTime;
}

void TimeManagement::startTimer() {
	_startTime = millis();
}

void TimeManagement::startState(int stateId) {
	Serial.print("Hi Start ");
	Serial.println(stateId);
	Serial.print("Time for task ");
	Serial.println(_stateTimeLimits[stateId]);
	
	if (stateId > 0 && stateId < 17) {
		TickType_t  stateTime = pdMS_TO_TICKS(_stateTimeLimits[stateId]);
		xTimerReset(StateTimeoutTimer, 0);
		xTimerChangePeriod(StateTimeoutTimer, stateTime, 0);
		xTimerReset(StateTimeoutTimer, 0);
		
		Serial.println("Start Timer");
		xTimerStart(StateTimeoutTimer, 0);
	}

	

	_stateStartTime = millis();
}


void TimeManagement::endState(int stateId,  bool allocate) {
	_stateEndTime = millis();
	if (stateId >= 0 && stateId < 17) {
		xTimerStop(StateTimeoutTimer, 0);
		xTimerReset(StateTimeoutTimer, 0);
	}
	Serial.print("Hi End ");
	Serial.println(stateId);
	if (stateId == 0) {
		startTimer();
	}
	long elapsedTime = _stateEndTime - _stateStartTime;
	Serial.print("Elapsed Time for Task: ");
	Serial.println(elapsedTime);
	Serial.print("Max Time for Task: ");
	Serial.println(_stateTimeLimits[stateId]);
	long addToBank = _stateTimeLimits[stateId] - elapsedTime;
	_timeBank += addToBank;
	if (allocate) {
		allocateTime(stateId);
	}
	if (stateId > 0 && stateId < 17) {
		Serial.println("Clearing Buffer");
		clearBuffer();	
	}
	

}



void TimeManagement::allocateTime(int stateId) {
	Serial.println("Hi Allocate");
	
	long rTime = getRemainingTime();
	Serial.print("Overall Remaining Time: ");
	Serial.print(rTime);
	Serial.println(" ms");

	if (_timeBank > 0) {
        // Calculate the total of the max times for all subsequent states
        unsigned long totalMaxTimeRemaining = 0;
        for (unsigned int i = stateId + 1; i < _numStates-2; ++i) {
            totalMaxTimeRemaining += _stateTimeLimits[i];
        }

        // Distribute the time bank to all subsequent states based on their max times
		while (_timeBank > 500) {
			for (int i = stateId + 1; i < _numStates-2; ++i) {
				if (totalMaxTimeRemaining > 0) {
					// Calculate the proportion of max time for each state relative to the total
					float proportion = (float)_stateTimeLimits[i] / totalMaxTimeRemaining;

					// Determine the amount of time to allocate to the state based on the proportion
					long extraTime = (long)(proportion * _timeBank);

					// Ensure we don't allocate more than what's available in the time bank
					extraTime = min(extraTime, _timeBank);
					long origlimit = _stateTimeLimits[i];
					// Allocate the extra time to the state
					_stateTimeLimits[i] += extraTime;
					
					
					long origBank = _timeBank;
					// Decrease the time bank by the allocated time
					_timeBank -= extraTime;
										
				}
			}
		}
    }
	Serial.print("Timebank: ");
	Serial.println(_timeBank);
}




long TimeManagement::getRemainingTimeForState(int stateId) const {
	// Return the remaining time for a specific state
	long stateCurrentTime = millis();

	long stateElapsedTime = stateCurrentTime - _stateStartTime;
	
	return _stateTimeLimits[stateId] - stateElapsedTime;
}

long TimeManagement::getRunTime() const {
	long currentTime = millis();
	long elTime = currentTime - _startTime;
	return elTime;
}


long TimeManagement::getRemainingTime() const {
	long currentTime = millis();
	long elTime = currentTime - _startTime;
	return _maxTime - elTime;
}

long TimeManagement::getTimeBank() const {
		Serial.println("Hi GetBank");
    return _timeBank;
}

long TimeManagement::getStateLimit(int stateId) const {
		
    return _stateTimeLimits[stateId];
}


void TimeManagement::StateTimeoutCallback(TimerHandle_t xTimer) {
	Serial.println("Timeout");
    TimeManagement* instance = static_cast<TimeManagement*>(pvTimerGetTimerID(xTimer));
    if (instance != nullptr) {
        Serial.println("Timeout");
        bool timeOut = true; // You might need to adjust this part based on your requirements
        uint32_t valueToSend = timeOut ? 1 : 0;
        xTaskNotify(instance->MotorBoxTaskHandle, valueToSend, eSetValueWithOverwrite);
    }
}