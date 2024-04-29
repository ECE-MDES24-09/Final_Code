//
// Created by jorda on 03/15/2024.
//
#include "gyro_red.h"

void gyro_init() {
    while (!SERIAL_PORT) {
    };

#ifdef USE_SPI
    SPI_PORT.begin();
#else
    WIRE_PORT.begin();
    WIRE_PORT.setClock(400000);
#endif

    myICM.enableDebugging(); // Uncomment this line to enable helpful debug messages on Serial

    bool initialized = false;
    while (!initialized) {
#ifdef USE_SPI
        myICM.begin(CS_PIN, SPI_PORT);
#else
        myICM.begin(WIRE_PORT, AD0_VAL);
#endif
        SERIAL_PORT.print(F("Initialization of the sensor returned: "));
        SERIAL_PORT.println(myICM.statusString());
        if (myICM.status != ICM_20948_Stat_Ok) {
            SERIAL_PORT.println("Trying again...");
            delay(500);
        } else {
            initialized = true;
        }
    }
}


float getGyroAng() {
    float test = 0;
    float x = 0;
    float y = 0;
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float x3 = 0;
    float y3 = 0;
    int16_t z = 0;
    float Angle = 0;


    if (myICM.dataReady()) {
        myICM.getAGMT(); // The values are only updated when you call 'getAGMT'
        //y = myICM.getAGMT().mag.axes.y;         // The values are only updated when you call 'getAGMT'
        x1 = myICM.magX() + 40;
        y1 = myICM.magY() - 70;
        myICM.getAGMT();
        x2 = myICM.magX() + 40;
        y2 = myICM.magY() - 70;
        myICM.getAGMT();
        x3 = myICM.magX() + 40;
        y3 = myICM.magY() - 70;

        x = (x1 + x2 + x3) / 3;
        y = (y1 + y2 + y3) / 3;

        Angle = atan2(x, y) * 180.0 / PI + 180;
        Serial.print(x);
        Serial.print(" ");
        Serial.println(y);
    }

    return Angle;
}