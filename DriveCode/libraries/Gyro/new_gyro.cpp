#include "new_gyro.h"

New_Gyro::Gyro() : mpu(Wire) {}

void New_Gyro::init() {
    Wire.begin();

	byte status = mpu.begin();
	Serial.print(F("MPU6050 status: "));
	Serial.println(status);
	while (status != 0) {} // stop everything if could not connect to MPU6050

	Serial.println(F("Calculating offsets, do not move MPU6050"));
	// mpu.upsideDownMounting = true; // uncomment this line if the MPU6050 is mounted upside-down
	mpu.calcOffsets(); // gyro and accelero
	Serial.println("Done!\n");
}

void New_Gyro::updateYawPitchRoll() {
    mpu.update();
}

void New_Gyro::printYawPitchRoll() {
    // Function implementation
}

float New_Gyro::getYaw() {
    return (mpu.getAngleX() * mult);
}
