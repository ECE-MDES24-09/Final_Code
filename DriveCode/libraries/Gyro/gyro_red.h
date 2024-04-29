#ifndef GYRO_RED_H
#define GYRO_RED_H

#include "ICM_20948.h" // Library for the ICM-20948 sensor

//#define USE_SPI       // Uncomment this to use SPI

#ifdef USE_SPI
#include <SPI.h>
ICM_20948_SPI myICM; // If using SPI, create an ICM_20948_SPI object
#else
#include <Wire.h>
ICM_20948_I2C myICM; // Otherwise, create an ICM_20948_I2C object
#endif

void gyro_init();
float getGyroAng();

#endif // GYRO_RED_H
