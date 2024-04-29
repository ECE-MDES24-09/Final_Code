#ifndef SERIAL_COMMUNICATION_H
#define SERIAL_COMMUNICATION_H

#define DATA_LENGTH 5
#define START_WORD "START"
#define START_WORD_LEN 5

#include "Arduino.h"

byte* getSerialData();

#endif