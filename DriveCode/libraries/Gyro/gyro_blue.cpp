//
// Created by jorda on 03/15/2024.
//
#include "gyro_blue.h"


void Gyro::init() {
    Wire.begin();
    delay(2000);

    if (!mpu.setup(0x68)) {
        Serial.println(F("MPU connection failed. Please check your connection with `connection_check` example."));
        while (1) {
            delay(5000);
        }
    }

    filter.begin(25); // If your filter library supports begin function then set the sample rate here

    if (CALIBRATE) {
        // calibrate anytime you want to
        Serial.println("Accel Gyro calibration will start in 5sec.");
        Serial.println("Please leave the device still on the flat plane.");
        mpu.verbose(true);
        delay(5000);
        mpu.calibrateAccelGyro();


        Serial.println("Mag calibration will start in 5sec.");
        Serial.println("Please Wave device in a figure eight until done.");
        delay(5000);
        mpu.calibrateMag();

        mpu.verbose(false);
    }
}

void Gyro::updateYawPitchRoll() {
    float ax, ay, az, gx, gy, gz, mx, my, mz;
    unsigned long microsNow;

    microsNow = micros();
    if (microsNow - microsPrevious >= microsPerReading) {
        if (mpu.update()) {
            ax = mpu.getAccX();
            ay = mpu.getAccY();
            az = mpu.getAccZ();
            gx = mpu.getGyroX();
            gy = mpu.getGyroY();
            gz = mpu.getGyroZ();
            mx = mpu.getMagX();
            my = mpu.getMagY();
            mz = mpu.getMagZ();

            filter.update(gx, gy, gz, ax, ay, az, mx, my, mz);
            // filter.updateIMU(gx, gy, gz, ax, ay, az);

            // Directly concatenate float values into the string to be printed.
            Serial.print(filter.getYaw());
            Serial.print(F(","));
            Serial.print(filter.getPitch());
            Serial.print(F(","));
            Serial.println(filter.getRoll());

            microsPrevious = microsNow;
        }
    }
}

float Gyro::getYaw() {
    updateYawPitchRoll();
    return filter.getYaw();
}