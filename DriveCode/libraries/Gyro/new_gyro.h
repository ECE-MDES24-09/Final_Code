#ifndef NEW_GYRO_H
#define NEW_GYRO_H

#include "Wire.h"
#include <MPU6050_light.h>

class Gyro {
public:
    void init();
    void updateYawPitchRoll();
    void printYawPitchRoll();
    float getYaw();

private:
    MPU6050 mpu;
    unsigned long timer;
    static const int mult = 180 / 3.14;
};

#endif // NEW_GYRO_H
