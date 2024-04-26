#include "printf.h"

#if defined (ARDUINO) && !defined (__arm__) && !defined(__ARDUINO_X86__)
int serial_putc(char c, FILE *) {
    Serial.write(c);
    return c;
}

void printf_begin(void) {
    fdevopen(&serial_putc, 0);
}
#elif defined (__arm__)
void printf_begin(void) {
    // Empty implementation for ARM based boards
}
#elif defined(__ARDUINO_X86__)
int serial_putc(char c, FILE *) {
    Serial.write(c);
    return c;
}

void printf_begin(void) {
    //For redirecting stdout to /dev/ttyGS0 (Serial Monitor port)
    stdout = freopen("/dev/ttyGS0", "w", stdout);
    delay(500);
    printf("redirecting to Serial...");
}
#endif
