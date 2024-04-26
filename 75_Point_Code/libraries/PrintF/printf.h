#ifndef __PRINTF_H__
#define __PRINTF_H__

#include "Arduino.h"

#if defined (ARDUINO) && !defined (__arm__) && !defined(__ARDUINO_X86__)
int serial_putc(char c, FILE *);
void printf_begin(void);
#elif defined (__arm__)
void printf_begin(void);
#elif defined(__ARDUINO_X86__)
int serial_putc(char c, FILE *);
void printf_begin(void);
#else
#error This example is only for use on Arduino.
#endif // ARDUINO

#endif // __PRINTF_H__
