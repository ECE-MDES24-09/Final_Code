#include "MPU9250.h"
#include <MadgwickAHRS.h>

#define CALIBRATE false

class Gyro {
public:
    void init();

    void updateYawPitchRoll();

    float getYaw();

private:
    MPU9250 mpu;
    Madgwick filter;
    const unsigned long microsPerReading = 1000000 / 25;
    unsigned long microsPrevious = 0;
};
