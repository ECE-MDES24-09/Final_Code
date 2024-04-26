#ifndef SERIAL_COMMUNICATION_H
#define SERIAL_COMMUNICATION_H

#define DATA_LENGTH 5
#define START_WORD "START"
#define START_WORD_LEN 5

#include "Arduino.h"

unsigned long lastTimeRecevieData = 0;

byte* getSerialData() {
    static byte data[DATA_LENGTH] = { 0, 0, 0 }; // Data buffer initialized to zeros
    static char buffer[128]; // Buffer to store incoming characters plus "START"
    static int bufferIndex = 0; // Current position in the buffer

    // Read all available characters from the serial port
    while (Serial1.available() > 0 && bufferIndex < 63) {
        char receivedChar = (char)Serial1.read(); // Read the next character
        buffer[bufferIndex++] = receivedChar; // Store it in the buffer
        buffer[bufferIndex] = '\0'; // Null-terminate the string

        // Look for "START" in the buffer
        char* found = strstr(buffer, "START");
        if (found) {  // If "START" found in the buffer
            int startOffset = found - buffer;  // Get the offset of "START" in the buffer
            int charsAfterStart = bufferIndex - startOffset - 5; // Characters after "START"

            if (charsAfterStart >= DATA_LENGTH) {
                // If at least three characters after "START", copy them to data
                memcpy(data, found + 5, DATA_LENGTH);
                // Move any excess characters to the beginning of the buffer
                bufferIndex = bufferIndex - (startOffset + 5 + DATA_LENGTH);
                memmove(buffer, &buffer[startOffset + 5 + DATA_LENGTH], bufferIndex);
                lastTimeRecevieData = millis();
                // Serial.println("working");
                return data; // Return the data
            }
        }

        // If buffer is close to full and no "START" has been processed, shift buffer
        if (bufferIndex > 56) { // Buffer length minus "START" length minus data length
            // Keep only the last part of the buffer that can potentially have "START"
            memmove(buffer, buffer + bufferIndex - 7, 7);
            bufferIndex = 7;
        }
    }

    // Serial.print("not working");
    // If we reach here, we haven't found a valid sequence, return null or keep searching
    static byte notData[DATA_LENGTH] = { 0, 0, 0, 0, 0};
    return notData; // Returning NULL signifies that no valid data is available
}



#endif